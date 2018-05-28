from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import tensorflow.contrib.keras.api.keras.backend as K
from scipy.fftpack import idct
from scipy.linalg import pinv
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Input
from tensorflow.contrib.keras.api.keras.layers import Conv2DTranspose, Conv2D, \
    Dense, Activation, Lambda, Add, Concatenate, Permute, Reshape, Flatten, \
    Conv1D, Multiply, Embedding, LeakyReLU, ZeroPadding2D, Cropping2D
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.regularizers import l2
from tensorflow.contrib.keras.api.keras.initializers import Constant
from layers.normalization import InstanceNormalization
from layers.edm import edm, EDM
from layers.comb_matrix import CombMatrix
from layers.tile import Tile
from layers.cudnn_recurrent import CuDNNLSTM
from collections import OrderedDict
from utils.scoping import Scoping
from utils.tfangles import quaternion_between, quaternion_to_expmap, expmap_to_rotmat, rotmat_to_euler, \
    vector3d_to_quaternion, quaternion_conjugate, rotate_vector_by_quaternion, rotmat_to_quaternion, \
    expmap_to_quaternion, quaternion_to_rotmat
from utils.seq_utils import get_body_graph

CONV1D_ARGS = {'padding': 'same', 'kernel_regularizer': l2(5e-4)}
CONV2D_ARGS = {'padding': 'same', 'data_format': 'channels_last', 'kernel_regularizer': l2(5e-4)}


def get_model(config):
    class_name = 'MotionGANV' + config.model_version[-1]
    module = __import__('models.motiongan', fromlist=[class_name])
    my_class = getattr(module, class_name)
    return my_class(config)


def _get_tensor(tensors, name):
    if isinstance(tensors, list):
        return next(obj for obj in tensors if name in obj.name)
    else:
        return tensors


class _MotionGAN(object):
    def __init__(self, config):
        self.name = config.model_type + '_' + config.model_version
        self.data_set = config.data_set
        self.batch_size = config.batch_size
        self.num_actions = config.num_actions
        self.seq_len = config.pick_num if config.pick_num > 0 else (
                       config.crop_len if config.crop_len > 0 else None)
        self.njoints = config.njoints
        self.body_members = config.body_members

        self.dropout = config.dropout
        self.lambda_grads = config.lambda_grads
        self.gamma_grads = 1.0
        self.wgan_scale_d = 10.0 * config.loss_factor
        self.wgan_scale_g = 2.0 * config.loss_factor * (0.0 if config.no_gan_loss else 1.0)
        self.wgan_frame_scale_d = 10.0 * config.loss_factor
        self.wgan_frame_scale_g = 2.0 * config.loss_factor * (0.0 if config.no_gan_loss else 1.0)
        self.rec_scale = 1.0   # if 'expmaps' not in self.data_set else 10.0
        self.action_cond = config.action_cond
        self.action_scale_d = 10.0
        self.action_scale_g = 1.0
        # self.latent_cond_dim = config.latent_cond_dim
        self.latent_scale_d = 10.0
        self.latent_scale_g = 1.0
        self.shape_loss = config.shape_loss
        self.shape_scale = 1.0
        self.smoothing_loss = config.smoothing_loss
        self.smoothing_scale = 20.0
        self.smoothing_basis = 5
        self.time_pres_emb = config.time_pres_emb
        self.use_pose_fae = config.use_pose_fae
        self.rotation_loss = config.rotation_loss
        self.rotation_scale = 10.0
        self.translate_start = config.translate_start
        self.rotate_start = config.rotate_start
        self.rescale_coords = config.rescale_coords
        self.remove_hip = config.remove_hip
        self.use_diff = config.use_diff
        self.diff_scale = 100.0
        self.use_angles = config.use_angles
        self.angles_scale = 0.5
        self.stats = {}

        # Discriminator
        true_label = Input(batch_shape=(self.batch_size, 1), name='true_label', dtype='int32')
        real_seq = Input(batch_shape=(self.batch_size, self.njoints, self.seq_len, 3), name='real_seq', dtype='float32')
        self.disc_inputs = [real_seq]
        self.place_holders = [true_label]  # it is not an input because it is only used in the loss
        x = self._proc_disc_inputs(self.disc_inputs)
        self.real_outputs = self._proc_disc_outputs(self.discriminator(x))
        self.disc_model = Model(self.disc_inputs, self.real_outputs, name=self.name + '_discriminator')

        # Generator
        seq_mask = Input(batch_shape=(self.batch_size, self.njoints, self.seq_len, 1), name='seq_mask', dtype='float32')
        self.gen_inputs = [real_seq, seq_mask]
        # if self.latent_cond_dim > 0:
        #     latent_cond_input = Input(batch_shape=(self.batch_size, self.latent_cond_dim),
        #                               name='latent_cond_input', dtype='float32')
        #     self.gen_inputs.append(latent_cond_input)
        if self.action_cond:
            self.gen_inputs.append(true_label)
        x = self._proc_gen_inputs(self.gen_inputs)
        self.gen_outputs = self._proc_gen_outputs(self.generator(x))
        self.gen_model = Model(self.gen_inputs, self.gen_outputs, name=self.name + '_generator')
        self.fake_outputs = self.disc_model(self.gen_outputs)

        # Losses
        self.wgan_losses, self.disc_losses, self.gen_losses = self._build_loss()

        with K.name_scope('loss/sum'):
            disc_loss = 0.0
            for loss in self.disc_losses.values():
                disc_loss += loss

            gen_loss = 0.0
            for loss in self.gen_losses.values():
                gen_loss += loss

        # Custom train functions
        with K.name_scope('discriminator/functions/train'):
            disc_optimizer = Adam(lr=config.learning_rate)
            disc_training_updates = disc_optimizer.get_updates(disc_loss, self.disc_model.trainable_weights)
            self.disc_train_f = K.function(self.disc_inputs + self.gen_inputs + self.place_holders,
                                           self.wgan_losses.values() + self.disc_losses.values(),
                                           disc_training_updates)

        with K.name_scope('discriminator/functions/eval'):
            self.disc_eval_f = K.function(self.disc_inputs + self.gen_inputs + self.place_holders,
                                          self.wgan_losses.values() + self.disc_losses.values())

        self.disc_model = self._pseudo_build_model(self.disc_model, disc_optimizer)

        with K.name_scope('generator/functions/train'):
            gen_optimizer = Adam(lr=config.learning_rate)
            gen_training_updates = gen_optimizer.get_updates(gen_loss, self.gen_model.trainable_weights)
            self.gen_train_f = K.function(self.gen_inputs + self.place_holders, self.gen_losses.values(), gen_training_updates)

        with K.name_scope('generator/functions/eval'):
            gen_f_outs = self.gen_losses.values()
            if self.use_pose_fae:
                gen_f_outs.append(self.fae_z)
            # gen_f_outs.append(self.aux_out)
            gen_f_outs += self.gen_outputs
            self.gen_eval_f = K.function(self.gen_inputs + self.place_holders, gen_f_outs)

        self.gen_model = self._pseudo_build_model(self.gen_model, gen_optimizer)

        # GAN, complete model
        self.gan_model = Model(self.gen_inputs,
                               self.disc_model(self.gen_model(self.gen_inputs)),
                               name=self.name + '_gan')

    def disc_train(self, inputs):
        train_outs = self.disc_train_f(inputs)
        keys = self.wgan_losses.keys() + self.disc_losses.keys()
        keys = ['train/%s' % key for key in keys]
        losses_dict = OrderedDict(zip(keys, train_outs))
        return losses_dict

    def disc_eval(self, inputs):
        eval_outs = self.disc_eval_f(inputs)
        keys = self.wgan_losses.keys() + self.disc_losses.keys()
        keys = ['val/%s' % key for key in keys]
        losses_dict = OrderedDict(zip(keys, eval_outs))
        return losses_dict

    def gen_train(self, inputs):
        train_outs = self.gen_train_f(inputs)
        keys = self.gen_losses.keys()
        keys = ['train/%s' % key for key in keys]
        losses_dict = OrderedDict(zip(keys, train_outs))
        return losses_dict

    def gen_eval(self, inputs):
        eval_outs = self.gen_eval_f(inputs)
        keys = self.gen_losses.keys()
        keys = ['val/%s' % key for key in keys]
        if self.use_pose_fae:
            keys.append('fae_z')
        # keys.append('aux_out')
        keys.append('gen_outputs')
        losses_dict = OrderedDict(zip(keys, eval_outs))
        return losses_dict

    def update_lr(self, lr):
        K.set_value(self.disc_model.optimizer.lr, lr)
        K.set_value(self.gen_model.optimizer.lr, lr)

    def _build_loss(self):
        with K.name_scope('loss'):
            # Dicts to store the losses
            wgan_losses = OrderedDict()
            disc_losses = OrderedDict()
            gen_losses = OrderedDict()

            # Grabbing tensors
            real_seq = _get_tensor(self.disc_inputs, 'real_seq')
            seq_mask = _get_tensor(self.gen_inputs, 'seq_mask')
            gen_seq = self.gen_outputs[0]

            no_zero_frames = K.cast(K.greater_equal(K.abs(K.sum(real_seq, axis=(1, 3))), K.epsilon()), 'float32')
            no_zero_frames_edm = K.reshape(no_zero_frames, (no_zero_frames.shape[0], 1, 1, no_zero_frames.shape[1]))

            # WGAN Basic losses
            with K.name_scope('wgan_loss'):
                loss_real = K.mean(_get_tensor(self.real_outputs, 'score_out'), axis=-1)
                loss_fake = K.mean(_get_tensor(self.fake_outputs, 'score_out'), axis=-1)
                wgan_losses['loss_real'] = K.mean(loss_real)
                wgan_losses['loss_fake'] = K.mean(loss_fake)

                # Interpolates for GP
                alpha = K.random_uniform((self.batch_size, 1, 1, 1))
                interpolates = (alpha * real_seq) + ((1 - alpha) * gen_seq)

                # Gradient Penalty
                inter_outputs = self.disc_model(interpolates)
                inter_score = _get_tensor(inter_outputs, 'score_out')
                grad_mixed = K.gradients(inter_score, [interpolates])[0]
                norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=(1, 2, 3)) + K.epsilon())
                grad_penalty = K.mean(K.square(norm_grad_mixed - self.gamma_grads) / (self.gamma_grads ** 2), axis=-1)

                # WGAN-GP losses
                disc_loss_wgan = loss_fake - loss_real + (self.lambda_grads * grad_penalty)
                disc_losses['disc_loss_wgan'] = self.wgan_scale_d * K.mean(disc_loss_wgan)

                gen_loss_wgan = -loss_fake
                gen_losses['gen_loss_wgan'] = self.wgan_scale_g * K.mean(gen_loss_wgan)

            with K.name_scope('frame_wgan_loss'):
                frame_loss_real = K.sum(K.squeeze(_get_tensor(self.real_outputs, 'frame_score_out'), axis=-1) * no_zero_frames, axis=1)
                frame_loss_fake = K.sum(K.squeeze(_get_tensor(self.fake_outputs, 'frame_score_out'), axis=-1) * no_zero_frames, axis=1)
                wgan_losses['frame_loss_real'] = K.mean(frame_loss_real)
                wgan_losses['frame_loss_fake'] = K.mean(frame_loss_fake)

                interpolates = (alpha * real_seq) + ((1 - alpha) * gen_seq)

                inter_outputs = self.disc_model(interpolates)
                frame_inter_score = _get_tensor(inter_outputs, 'frame_score_out')
                frame_grad_mixed = K.gradients(frame_inter_score, [interpolates])[0]
                frame_norm_grad_mixed = K.sqrt(K.sum(K.sum(K.square(frame_grad_mixed), axis=(1, 3)) * no_zero_frames, axis=1) + K.epsilon())
                frame_grad_penalty = K.mean(K.square(frame_norm_grad_mixed - self.gamma_grads) / (self.gamma_grads ** 2), axis=-1)

                frame_disc_loss_wgan = frame_loss_fake - frame_loss_real + (self.lambda_grads * frame_grad_penalty)
                disc_losses['frame_disc_loss_wgan'] = self.wgan_frame_scale_d * K.mean(frame_disc_loss_wgan)

                frame_gen_loss_wgan = -frame_loss_fake
                gen_losses['frame_gen_loss_wgan'] = self.wgan_frame_scale_g * K.mean(frame_gen_loss_wgan)

            # Reconstruction loss
            with K.name_scope('reconstruction_loss'):
                loss_rec = K.sum(K.mean(K.square((real_seq * seq_mask) - (gen_seq * seq_mask)), axis=-1), axis=(1, 2))
                gen_losses['gen_loss_rec'] = self.rec_scale * K.mean(loss_rec)

                if self.use_diff:
                    loss_rec_diff = K.sum(K.mean(K.square((self.diff_input * self.diff_mask) -
                                                          (self.diff_output * self.diff_mask)), axis=-1), axis=(1, 2))
                    gen_losses['gen_loss_rec_diff'] = self.diff_scale * K.mean(loss_rec_diff)
                if self.use_angles:
                    loss_rec_angles = K.sum(K.mean(K.square((self.angles_input * self.angles_mask) -
                                                            (self.angles_output * self.angles_mask)), axis=-1), axis=(1, 2))
                    gen_losses['gen_loss_rec_angles'] = self.angles_scale * K.mean(loss_rec_angles)

            # Action label loss
            with K.name_scope('action_loss'):
                loss_class_real = K.mean(K.sparse_categorical_crossentropy(
                    _get_tensor(self.place_holders, 'true_label'),
                    _get_tensor(self.real_outputs, 'label_out'), True))
                loss_class_fake = K.mean(K.sparse_categorical_crossentropy(
                    _get_tensor(self.place_holders, 'true_label'),
                    _get_tensor(self.fake_outputs, 'label_out'), True))
                disc_losses['disc_loss_action'] = self.action_scale_d * (loss_class_real + loss_class_fake)
                gen_losses['gen_loss_action'] = self.action_scale_g * loss_class_fake

            # Optional losses
            # if self.latent_cond_dim > 0:
            #     with K.name_scope('latent_loss'):
            #         loss_latent = K.mean(K.square(_get_tensor(self.fake_outputs, 'latent_cond_out')
            #                                       - _get_tensor(self.gen_inputs, 'latent_cond_input')))
            #         disc_losses['disc_loss_latent'] = self.latent_scale_d * loss_latent
            #         gen_losses['gen_loss_latent'] = self.latent_scale_g * loss_latent
            if self.shape_loss:
                with K.name_scope('shape_loss'):
                    mask = np.zeros((self.njoints, self.njoints), dtype='float32')
                    for member in self.body_members.values():
                        for j in range(len(member['joints']) - 1):
                            mask[member['joints'][j], member['joints'][j + 1]] = 1.0
                            mask[member['joints'][j + 1], member['joints'][j]] = 1.0
                    mask = np.reshape(mask, (1, self.njoints, self.njoints, 1))
                    mask = K.constant(mask, dtype='float32')
                    real_shape = K.sum(edm(real_seq) * no_zero_frames_edm, axis=-1, keepdims=True) \
                                 / K.sum(no_zero_frames_edm, axis=-1, keepdims=True) * mask
                    gen_shape = edm(gen_seq) * no_zero_frames_edm * mask
                    loss_shape = K.sum(K.square(real_shape - gen_shape), axis=(1, 2, 3))
                    gen_losses['gen_loss_shape'] = self.shape_scale * K.mean(loss_shape)

                    head_top = self.body_members['head']['joints'][-1]
                    left_hand = self.body_members['left_arm']['joints'][-1]
                    right_hand = self.body_members['right_arm']['joints'][-1]
                    left_foot = self.body_members['left_leg']['joints'][-1]
                    right_foot = self.body_members['right_leg']['joints'][-1]

                    mask = np.zeros((self.njoints, self.njoints), dtype='float32')
                    for j in [head_top, left_hand, right_hand, left_foot, right_foot]:
                        mask[j, :] = 1.0
                        mask[:, j] = 1.0

                    mask = np.reshape(mask, (1, self.njoints, self.njoints, 1))
                    mask = K.constant(mask, dtype='float32')
                    seq_mask_edm = K.prod(K.expand_dims(seq_mask, axis=1) * K.expand_dims(seq_mask, axis=2), axis=-1)
                    real_shape = edm(real_seq) * seq_mask_edm * mask
                    gen_shape = edm(gen_seq) * seq_mask_edm * mask
                    loss_shape = K.sum(K.square(real_shape - gen_shape), axis=(1, 2, 3))
                    gen_losses['gen_loss_limbs'] = self.shape_scale * K.mean(loss_shape)

            if self.rotation_loss:
                with K.name_scope('rotation_loss'):
                    def vector_mag(x):
                        return K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True) + K.epsilon())
                    masked_real = real_seq * seq_mask
                    masked_gen = gen_seq * seq_mask
                    unit_real = masked_real / vector_mag(masked_real)
                    unit_gen = masked_gen / vector_mag(masked_gen)
                    unit_real = K.reshape(unit_real, [-1, 3])
                    unit_gen = K.reshape(unit_gen, [-1, 3])
                    loss_rot = K.square(1 - K.batch_dot(unit_real, unit_gen, axes=[-1, -2]))
                    gen_losses['gen_loss_rotation'] = self.rotation_scale * K.mean(loss_rot)
            if self.smoothing_loss:
                with K.name_scope('smoothing_loss'):
                    Q = idct(np.eye(self.seq_len))[:self.smoothing_basis, :]
                    Q_inv = pinv(Q)
                    Qs = K.constant(np.matmul(Q_inv, Q))
                    gen_seq_s = K.permute_dimensions(gen_seq, (0, 1, 3, 2))
                    gen_seq_s = K.dot(gen_seq_s, Qs)
                    gen_seq_s = K.permute_dimensions(gen_seq_s, (0, 1, 3, 2))
                    loss_smooth = K.sum(K.mean(K.square(gen_seq_s - gen_seq), axis=-1), axis=(1, 2))
                    gen_losses['gen_loss_smooth'] = self.smoothing_scale * K.mean(loss_smooth)


            # Regularization losses
            with K.name_scope('regularization_loss'):
                if len(self.disc_model.losses) > 0:
                    disc_loss_reg = 0.0
                    for reg_loss in set(self.disc_model.losses):
                        disc_loss_reg += reg_loss
                    disc_losses['disc_loss_reg'] = disc_loss_reg

                if len(self.gen_model.losses) > 0:
                    gen_loss_reg = 0.0
                    for reg_loss in set(self.gen_model.losses):
                        gen_loss_reg += reg_loss
                    gen_losses['gen_loss_reg'] = gen_loss_reg

        return wgan_losses, disc_losses, gen_losses

    def _pseudo_build_model(self, model, optimizer):
        # This function mimics compilation to enable saving the model
        model.optimizer = optimizer
        model.sample_weight_mode = None
        model.loss = 'custom_loss'
        model.loss_weights = None
        model.metrics = None
        return model

    def _remove_hip_in(self, x, x_mask):
        scope = Scoping.get_global_scope()
        with scope.name_scope('remove_hip'):

            if 'expmaps' in self.data_set:
                self.stats[scope+'hip_expmaps'] = Lambda(lambda arg: arg[:, :2, :, :], name=scope+'hip_expmaps')(x)

                x = Lambda(lambda arg: arg[:, 2:, ...], name=scope+'remove_hip_in')(x)
                x_mask = Lambda(lambda arg: arg[:, 2:, ...], name=scope+'remove_hip_mask_in')(x_mask)
            else:
                def _get_hips(arg):
                    return K.reshape(arg[:, 0, :, :], (arg.shape[0], 1, arg.shape[2], 3))

                self.stats[scope+'hip_coords'] = Lambda(_get_hips, name=scope+'hip_coords')(x)

                x = Lambda(lambda args: (args[0] - args[1])[:, 1:, ...],
                           name=scope+'remove_hip_in')([x, self.stats[scope+'hip_coords']])
                x_mask = Lambda(lambda arg: arg[:, 1:, ...], name=scope+'remove_hip_mask_in')(x_mask)
        return x, x_mask

    def _remove_hip_out(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('remove_hip'):

            if 'expmaps' in self.data_set:
                x = Lambda(lambda args: K.concatenate([args[1], args[0]], axis=1),
                           name=scope+'remove_hip_out')([x, self.stats[scope+'hip_expmaps']])
            else:
                x = Lambda(lambda args: K.concatenate([args[1], args[0] + args[1]], axis=1),
                           name=scope+'remove_hip_out')([x, self.stats[scope+'hip_coords']])
        return x

    def _translate_start_in(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('translate_start'):

            def _get_start(arg):
                return K.reshape(arg[:, 0, 0, :], (arg.shape[0], 1, 1, 3))

            self.stats[scope+'start_pt'] = Lambda(_get_start, name=scope+'start_pt')(x)

            x = Lambda(lambda args: args[0] - args[1], name=scope+'translate_start_in')([x, self.stats[scope+'start_pt']])
        return x

    def _translate_start_out(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('translate_start'):

            x = Lambda(lambda args: args[0] + args[1], name=scope+'translate_start_out')([x, self.stats[scope+'start_pt']])
        return x

    def _rotate_start_in(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('rotate_start'):

            left_shoulder = self.body_members['left_arm']['joints'][1]
            right_shoulder = self.body_members['right_arm']['joints'][1]
            hip = self.body_members['torso']['joints'][0]
            head_top = self.body_members['head']['joints'][-1]

            base_shape = [int(d) for d in x.shape]
            base_shape[1] = 1
            base_shape[2] = 1

            def _get_rotation(arg):
                coords_list = tf.unstack(arg[:, :, 0, :], axis=1)
                torso_rot = tf.cross(coords_list[left_shoulder] - coords_list[hip],
                                     coords_list[right_shoulder] - coords_list[hip])
                side_rot = K.reshape(tf.cross(coords_list[head_top] - coords_list[hip], torso_rot), base_shape)
                theta_diff = ((np.pi / 2) - tf.atan2(side_rot[..., 1], side_rot[..., 0])) / 2
                cos_theta_diff = tf.cos(theta_diff)
                sin_theta_diff = tf.sin(theta_diff)
                zeros_theta = K.zeros_like(sin_theta_diff)
                return K.stack([cos_theta_diff, zeros_theta, zeros_theta, sin_theta_diff], axis=-1)

            self.stats[scope+'start_rotation'] = Lambda(_get_rotation, name=scope+'start_rotation')(x)

            x = Lambda(lambda args: rotate_vector_by_quaternion(args[1], args[0]),
                       name=scope+'rotate_start_in')([x, self.stats[scope+'start_rotation']])
        return x

    def _rotate_start_out(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('rotate_start'):

            x = Lambda(lambda args: rotate_vector_by_quaternion(quaternion_conjugate(args[1]), args[0]),
                       name=scope+'rotate_start_out')([x, self.stats[scope+'start_rotation']])
        return x

    def _rescale_in(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('rescale'):

            members_from, members_to, _ = get_body_graph(self.body_members)

            def _get_avg_bone_len(arg):
                bone_list = tf.unstack(arg[:, :, 0, :], axis=1)
                bones = [bone_list[j] - bone_list[i] for i, j in zip(members_from, members_to)]
                bones = K.expand_dims(K.stack(bones, axis=1), axis=2)
                bone_len = K.sqrt(K.sum(K.square(bones), axis=-1, keepdims=True) + K.epsilon())
                return K.mean(bone_len, axis=1, keepdims=True)

            self.stats[scope+'bone_len'] = Lambda(_get_avg_bone_len, name=scope+'bone_len')(x)

            x = Lambda(lambda args: args[0] / args[1], name=scope+'rescale_in')([x, self.stats[scope+'bone_len']])
        return x

    def _rescale_out(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('rescale'):

            x = Lambda(lambda args: args[0] * args[1], name=scope+'rescale_out')([x, self.stats[scope+'bone_len']])
        return x

    def _seq_to_diff_in(self, x, x_mask=None):
        scope = Scoping.get_global_scope()
        with scope.name_scope('seq_to_diff'):

            self.stats[scope+'start_pose'] = Lambda(lambda arg: arg[:, :, 0, :], name=scope+'start_pose')(x)

            x = Lambda(lambda arg: arg[:, :, 1:, :] - arg[:, :, :-1, :], name=scope+'seq_to_diff_in')(x)

            if x_mask is not None:
                x_mask = Lambda(lambda arg: arg[:, :, 1:, :] * arg[:, :, :-1, :], name=scope+'seq_mask_to_diff_in')(x_mask)
        return x, x_mask

    def _seq_to_diff_out(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('seq_to_diff'):

            def _diff_to_seq(args):
                diffs, start_pose = args
                diffs_list = tf.unstack(diffs, axis=2)
                poses = [start_pose]
                for p in range(diffs.shape[2]):
                    poses.append(poses[p] + diffs_list[p])
                return K.stack(poses, axis=2)

            x = Lambda(_diff_to_seq, name=scope+'seq_to_diff_out')([x, self.stats[scope+'start_pose']])
        return x

    def _seq_to_angles_in(self, x, x_mask):
        scope = Scoping.get_global_scope()
        with scope.name_scope('seq_to_angles'):

            members_from, members_to, body_graph = get_body_graph(self.body_members)

            def _get_hips(arg):
                return K.reshape(arg[:, 0, :, :], (arg.shape[0], 1, arg.shape[2], 3))
            self.stats[scope+'hip_coords'] = Lambda(_get_hips, name=scope+'hip_coords')(x)

            def _get_bone_len(arg):
                bone_list = tf.unstack(arg[:, :, 0, :], axis=1)
                bones = [bone_list[j] - bone_list[i] for i, j in zip(members_from, members_to)]
                bones = K.stack(bones, axis=1)
                return K.sqrt(K.sum(K.square(bones), axis=-1) + K.epsilon())

            self.stats[scope+'bone_len'] = Lambda(_get_bone_len, name=scope+'bone_len')(x)

            def _get_angles(coords):
                base_shape = [int(dim) for dim in coords.shape]
                base_shape.pop(1)
                base_shape[-1] = 1

                coords_list = tf.unstack(coords, axis=1)

                def _get_angle_for_joint(joint_idx, parent_idx, angles):
                    if parent_idx is None:  # joint_idx should be 0
                        parent_bone = K.constant(np.concatenate([np.ones(base_shape),
                                                                 np.zeros(base_shape),
                                                                 np.zeros(base_shape)], axis=-1))
                    else:
                        parent_bone = coords_list[parent_idx] - coords_list[joint_idx]

                    for child_idx in body_graph[joint_idx]:
                        child_bone = coords_list[child_idx] - coords_list[joint_idx]
                        angle = quaternion_between(parent_bone, child_bone)
                        angle = quaternion_to_expmap(angle)
                        angles.append(angle)

                    for child_idx in body_graph[joint_idx]:
                        angles = _get_angle_for_joint(child_idx, joint_idx, angles)

                    return angles

                angles = _get_angle_for_joint(0, None, [])
                return K.stack(angles, axis=1)

            x = Lambda(_get_angles, name=scope+'angles')(x)

            def _get_angles_mask(coord_masks):
                base_shape = [int(dim) for dim in coord_masks.shape]
                base_shape.pop(1)
                base_shape[-1] = 1

                coord_masks_list = tf.unstack(coord_masks, axis=1)

                def _get_angle_mask_for_joint(joint_idx, angles_mask):
                    for child_idx in body_graph[joint_idx]:
                        angles_mask.append(coord_masks_list[child_idx])  # * coord_masks_list[joint_idx]

                    for child_idx in body_graph[joint_idx]:
                        angles_mask = _get_angle_mask_for_joint(child_idx, angles_mask)

                    return angles_mask

                angles_mask = _get_angle_mask_for_joint(0, [])
                return K.stack(angles_mask, axis=1)

            x_mask = Lambda(_get_angles_mask, name=scope+'angles_mask')(x_mask)

            fixed_angles = len(body_graph[0])
            self.stats[scope+'fixed_angles'] = Lambda(lambda args: args[:, :fixed_angles, ...], name=scope+'fixed_angles')(x)
            x = Lambda(lambda args: args[:, fixed_angles:, ...], name=scope+'motion_angles')(x)
            x_mask = Lambda(lambda args: args[:, fixed_angles:, ...], name=scope+'motion_angles_mask')(x_mask)

        return x, x_mask

    def _seq_to_angles_out(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('seq_to_angles'):

            members_from, members_to, body_graph = get_body_graph(self.body_members)

            x = Lambda(lambda args: K.concatenate(args, axis=1), name=scope+'concat_angles')(
                [self.stats[scope+'fixed_angles'], x])

            x = Lambda(lambda arg: expmap_to_rotmat(arg), name=scope+'rotmat')(x)
            self.euler_out = Lambda(lambda arg: rotmat_to_euler(arg), name=scope+'euler')(x)

            def _get_coords(args):
                rotmat, bone_len = args
                rotmat_list = tf.unstack(rotmat, axis=1)
                bone_len_list = tf.unstack(bone_len, axis=1)

                base_shape = [int(d) for d in rotmat.shape]
                base_shape.pop(1)
                base_shape[-2] = 1
                base_shape[-1] = 1
                bone_idcs = {idx_tup: i for i, idx_tup in enumerate([idx_tup for idx_tup in zip(members_from, members_to)])}
                trans_dims = range(len(base_shape))
                trans_dims[-1], trans_dims[-2] = trans_dims[-2], trans_dims[-1]

                def _get_coords_for_joint(joint_idx, parent_idx, child_angle_idx, coords):
                    if parent_idx is None:  # joint_idx should be 0
                        coords[joint_idx] = K.zeros(base_shape[:-2] + [3, 1])
                        parent_bone = K.constant(np.concatenate([np.ones(base_shape),
                                                                 np.zeros(base_shape),
                                                                 np.zeros(base_shape)], axis=-2))
                    else:
                        parent_bone = coords[parent_idx] - coords[joint_idx]
                        parent_bone_norm = K.sqrt(K.sum(K.square(parent_bone), axis=-2, keepdims=True) + K.epsilon())
                        parent_bone = parent_bone / parent_bone_norm

                    for child_idx in body_graph[joint_idx]:
                        child_bone = K.batch_dot(tf.transpose(rotmat_list[child_angle_idx], trans_dims), parent_bone , axes=[-1, -2])
                        child_bone_idx = bone_idcs[(joint_idx, child_idx)]
                        child_bone = child_bone * K.reshape(bone_len_list[child_bone_idx], (child_bone.shape[0], 1, 1, 1))
                        coords[child_idx] = child_bone + coords[joint_idx]
                        child_angle_idx += 1

                    for child_idx in body_graph[joint_idx]:
                        child_angle_idx, coords = _get_coords_for_joint(child_idx, joint_idx, child_angle_idx, coords)

                    return child_angle_idx, coords

                child_angle_idx, coords = _get_coords_for_joint(0, None, 0, {})
                coords = K.stack([t for i, t in sorted(coords.iteritems())], axis=1)
                coords = K.squeeze(coords, axis=-1)
                return coords

            x = Lambda(_get_coords, name=scope+'coords')([x, self.stats[scope+'bone_len']])
            x = Lambda(lambda args: args[0] + args[1], name=scope+'add_hip_coords')([x, self.stats[scope+'hip_coords']])
        return x

    def _proc_disc_inputs(self, input_tensors):
        scope = Scoping.get_global_scope()
        with scope.name_scope('discriminator'):

            x = _get_tensor(input_tensors, 'real_seq')

            if self.translate_start:
                x = self._translate_start_in(x)
            if self.rotate_start:
                x = self._rotate_start_in(x)
            if self.rescale_coords:
                x = self._rescale_in(x)

            self.org_shape = [int(dim) for dim in x.shape]

        return x

    def _proc_disc_outputs(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('discriminator'):
            score_out = Dense(1, name=scope+'score_out')(x)

            output_tensors = [score_out]

            label_out = Dense(self.num_actions, name=scope+'label_out')(x)
            output_tensors.append(label_out)

            z = self._pose_encoder(self.disc_inputs[0])

            frame_score_out = Conv1D(1, 1, 1, name=scope+'frame_score_out', **CONV1D_ARGS)(z)
            output_tensors.append(frame_score_out)

            # if self.latent_cond_dim > 0:
            #     latent_cond_out = Dense(self.latent_cond_dim, name=scope+'latent_cond_out')(x)
            #     output_tensors.append(latent_cond_out)

        return output_tensors

    def _proc_gen_inputs(self, input_tensors):
        scope = Scoping.get_global_scope()
        with scope.name_scope('generator'):

            x = _get_tensor(input_tensors, 'real_seq')
            x_mask = _get_tensor(input_tensors, 'seq_mask')

            if self.translate_start:
                x = self._translate_start_in(x)
            if self.rotate_start:
                x = self._rotate_start_in(x)
                # self.aux_out = x  # Uncomment to visualize rotated sequence
                # self.aux_out = self._rotate_start_out(x)  # Uncomment to visualize re-rotated sequence
            if self.rescale_coords:
                x = self._rescale_in(x)
            if self.remove_hip:
                x, x_mask = self._remove_hip_in(x, x_mask)
            if self.use_diff:
                x, x_mask = self._seq_to_diff_in(x, x_mask)
                self.diff_input, self.diff_mask = x, x_mask
            if self.use_angles:
                x, x_mask = self._seq_to_angles_in(x, x_mask)
                self.angles_input, self.angles_mask = x, x_mask
                # self.aux_out = self._seq_to_angles_out(x)  # Uncomment to visualize reconstructed sequence

            self.org_shape = [int(dim) for dim in x.shape]

            x = Multiply(name=scope+'mask_mult')([x, x_mask])
            x_occ = Lambda(lambda arg: 1 - arg, name=scope+'mask_occ')(x_mask)
            x = Concatenate(axis=-1, name=scope+'cat_occ')([x, x_occ])

            if self.use_pose_fae:

                self.fae_z = self._pose_encoder(x)
                x = Reshape((int(self.fae_z.shape[1]), int(self.fae_z.shape[2]), 1), name=scope+'gen_reshape_in')(self.fae_z)

                self.nblocks = 4

            else:
                with scope.name_scope('seq_fex'):
                    n_hidden = 32 if self.time_pres_emb else 128
                    strides = (2, 1) if self.time_pres_emb else 2
                    i = 0
                    while (x.shape[1] > 1 and self.time_pres_emb) or (i < 3):
                        with scope.name_scope('block_%d' % i):
                            num_block = n_hidden * (((i + 1) // 2) + 1)
                            shortcut = Conv2D(num_block, 1, strides,
                                              name=scope+'shortcut', **CONV2D_ARGS)(x)
                            with scope.name_scope('branch_0'): # scope for backward compat
                                pi = _conv_block(x, num_block, 8, 3, strides)
                            x = Add(name=scope+'add')([shortcut, pi])
                            x = Activation('relu', name=scope+'relu_out')(x)
                            i += 1

                    self.nblocks = i

                    if not self.time_pres_emb:
                        x = Lambda(lambda x: K.mean(x, axis=(1, 2)), name=scope+'mean_pool')(x)

            if self.action_cond:
                x_label = _get_tensor(input_tensors, 'true_label')
                x_label = Embedding(self.num_actions, 4, name=scope+'emb_label')(x_label)
                x_label = Reshape((1, 1, 4), name=scope+'res_label')(x_label)
                x_label = Tile((x.shape[1], x.shape[2], 1), name=scope+'tile_label')(x_label)
                x = Concatenate(axis=-1, name=scope+'cat_label')([x, x_label])


            # if self.latent_cond_dim > 0:
            #     x_lat = _get_tensor(input_tensors, 'latent_cond_input')
            #     x.append(x_lat)

        return x

    def _proc_gen_outputs(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('generator'):

            if self.use_pose_fae:
                x = Conv2D(1, 3, 1, name=scope+'fae_merge', **CONV2D_ARGS)(x)
                self.fae_gen_z = Reshape((int(x.shape[1]), int(x.shape[2])), name=scope+'fae_reshape')(x)

                x = self._pose_decoder(self.fae_gen_z)

            else:
                x = Permute((3, 2, 1), name=scope+'joint_permute')(x)  # filters, time, joints
                x = Conv2D(self.org_shape[1], 3, 1, name=scope+'joint_reshape', **CONV2D_ARGS)(x)
                x = Permute((1, 3, 2), name=scope+'time_permute')(x)  # filters, joints, time
                x = Conv2D(self.org_shape[2], 3, 1, name=scope+'time_reshape', **CONV2D_ARGS)(x)
                x = Permute((2, 3, 1), name=scope+'coords_permute')(x)  # joints, time, filters
                x = Conv2D(self.org_shape[3], 3, 1, name=scope+'coords_reshape', **CONV2D_ARGS)(x)

            if self.use_angles:
                self.angles_output = x
                x = self._seq_to_angles_out(x)
            if self.use_diff:
                self.diff_output = x
                x = self._seq_to_diff_out(x)
            if self.remove_hip:
                x = self._remove_hip_out(x)
            if self.rescale_coords:
                x = self._rescale_out(x)
            if self.rotate_start:
                x = self._rotate_start_out(x)
            if self.translate_start:
                x = self._translate_start_out(x)

            output_tensors = [x]

        return output_tensors

    def _pose_encoder(self, seq):
        scope = Scoping.get_global_scope()
        with scope.name_scope('encoder'):
            fae_dim = self.org_shape[1] * self.org_shape[3] * 2

            h = Permute((2, 1, 3), name=scope+'perm_in')(seq)
            h = Reshape((int(seq.shape[2]), int(seq.shape[1] * seq.shape[3])), name=scope+'resh_in')(h)

            h = Conv1D(fae_dim, 1, 1,
                       name=scope+'conv_in', **CONV1D_ARGS)(h)
            # self.pose_features = []
            for i in range(3):
                with scope.name_scope('block_%d' % i):
                    # self.pose_features.append(h)
                    pi = Conv1D(fae_dim, 1, 1, activation='relu',
                                name=scope+'pi_0', **CONV1D_ARGS)(h)
                    pi = Conv1D(fae_dim, 1, 1, activation='relu',
                                name=scope+'pi_1', **CONV1D_ARGS)(pi)
                    tau = Conv1D(fae_dim, 1, 1, activation='sigmoid',
                                 name=scope+'tau_0', **CONV1D_ARGS)(h)
                    h = Lambda(lambda args: (args[0] * (1 - args[2])) + (args[1] * args[2]),
                               name=scope+'attention')([h, pi, tau])

            z = Conv1D(fae_dim, 1, 1, name=scope+'z_mean', **CONV1D_ARGS)(h)
            z_attention = Conv1D(fae_dim, 1, 1, activation='sigmoid',
                                 name=scope+'attention_mask', **CONV1D_ARGS)(h)

            # We are only expecting half of the latent features to be activated
            z = Multiply(name=scope+'z_attention')([z, z_attention])

        return z

    def _pose_decoder(self, gen_z):
        scope = Scoping.get_global_scope()
        with scope.name_scope('decoder'):
            fae_dim = self.org_shape[1] * self.org_shape[3] * 2

            dec_h = Conv1D(fae_dim, 1, 1,
                           name=scope+'conv_in', **CONV1D_ARGS)(gen_z)
            for i in range(3):
                with scope.name_scope('block_%d' % i):
                    # dec_h = Concatenate(axis=-1, name=scope+'cat_feats')([dec_h, self.pose_features.pop()])
                    # dec_h = Conv1D(fae_dim * 2, 1, 1, activation='relu',
                    #                name=scope+'conv_h_0', **CONV1D_ARGS)(dec_h)
                    # dec_h = Conv1D(fae_dim, 1, 1, activation='relu',
                    #                name=scope+'conv_h_1', **CONV1D_ARGS)(dec_h)
                    pi = Conv1D(fae_dim, 1, 1, activation='relu',
                                name=scope+'pi_0', **CONV1D_ARGS)(dec_h)
                    pi = Conv1D(fae_dim, 1, 1, activation='relu',
                                name=scope+'pi_1', **CONV1D_ARGS)(pi)
                    tau = Conv1D(fae_dim, 1, 1, activation='sigmoid',
                                 name=scope+'tau_0', **CONV1D_ARGS)(dec_h)
                    dec_h = Lambda(lambda args: (args[0] * (1 - args[2])) + (args[1] * args[2]),
                                   name=scope+'attention')([dec_h, pi, tau])

            dec_x = Conv1D(self.org_shape[1] * 3, 1, 1, name=scope+'conv_out', **CONV1D_ARGS)(dec_h)
            dec_x = Reshape((int(gen_z.shape[1]), self.org_shape[1], 3), name=scope+'resh_out')(dec_x)
            dec_x = Permute((2, 1, 3), name=scope+'perm_out')(dec_x)

        return dec_x


def _conv_block(x, out_filters, bneck_factor, kernel_size, strides, conv_func=Conv2D, dilation_rate=(1, 1)):
    scope = Scoping.get_global_scope()
    # if 'generator' in str(scope):
    #     x = InstanceNormalization(axis=-1, name=scope+'inorm_in')(x)
    x = Activation('relu', name=scope+'relu_in')(x)
    x = conv_func(filters=out_filters // bneck_factor,
                  kernel_size=kernel_size, strides=1,
                  dilation_rate=dilation_rate, name=scope+'conv_in', **CONV2D_ARGS)(x)
    # if 'generator' in str(scope):
    #     x = InstanceNormalization(axis=-1, name=scope+'inorm_out')(x)
    x = Activation('relu', name=scope+'relu_out')(x)
    x = conv_func(filters=out_filters, kernel_size=kernel_size, strides=strides,
                  dilation_rate=dilation_rate, name=scope+'conv_out', **CONV2D_ARGS)(x)
    return x


class MotionGANV1(_MotionGAN):
    # ResNet

    def discriminator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('discriminator'):
            n_hidden = 64
            block_factors = [1, 1, 2, 2]
            block_strides = [2, 2, 1, 1]

            x = Conv2D(n_hidden * block_factors[0], 3, 1, name=scope+'conv_in', **CONV2D_ARGS)(x)
            for i, factor in enumerate(block_factors):
                with scope.name_scope('block_%d' % i):
                    n_filters = n_hidden * factor
                    shortcut = Conv2D(n_filters, block_strides[i], block_strides[i],
                                      name=scope+'shortcut', **CONV2D_ARGS)(x)
                    with scope.name_scope('branch_0'): # scope for backward compat
                        pi = _conv_block(x, n_filters, 1, 3, block_strides[i])

                    x = Add(name=scope+'add')([shortcut, pi])

            x = Activation('relu', name=scope+'relu_out')(x)
            x = Lambda(lambda x: K.mean(x, axis=(1, 2)), name=scope+'mean_pool')(x)

        return x

    def generator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('generator'):
            n_hidden = 32
            block_factors = range(1, self.nblocks + 1)
            block_strides = [2] * self.nblocks

            if not (self.time_pres_emb or self.use_pose_fae):
                x = Dense(4 * 4 * n_hidden * block_factors[0], name=scope+'dense_in')(x)
                x = Reshape((4, 4, n_hidden * block_factors[0]), name=scope+'reshape_in')(x)

            for i, factor in enumerate(block_factors):
                with scope.name_scope('block_%d' % i):
                    n_filters = n_hidden * factor
                    strides = block_strides[i]
                    if self.time_pres_emb:
                        strides = (block_strides[i], 1)
                    elif self.use_pose_fae:
                        strides = 1

                    with scope.name_scope('branch_0'): # scope for backward compat
                        pi = _conv_block(x, n_filters, 1, 3, strides, Conv2DTranspose)

                    if i < self.nblocks - 1:
                        shortcut = Conv2DTranspose(n_filters, strides, strides,
                                                   name=scope+'shortcut', **CONV2D_ARGS)(x)
                        x = Add(name=scope+'add')([shortcut, pi])
                    else:
                        x = pi

            # x = InstanceNormalization(axis=-1, name=scope+'inorm_out')(x)
            # x = Activation('relu', name=scope+'relu_out')(x)

        return x


class MotionGANV2(_MotionGAN):
    # Gated ResNet

    def discriminator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('discriminator'):
            n_hidden = 64
            block_factors = [1, 1, 2, 2]
            block_strides = [2, 2, 1, 1]

            x = Conv2D(n_hidden * block_factors[0], 3, 1, name=scope+'conv_in', **CONV2D_ARGS)(x)
            for i, factor in enumerate(block_factors):
                with scope.name_scope('block_%d' % i):
                    n_filters = n_hidden * factor
                    shortcut = Conv2D(n_filters, block_strides[i], block_strides[i],
                                      name=scope+'shortcut', **CONV2D_ARGS)(x)

                    with scope.name_scope('branch_0'):
                        pi = _conv_block(x, n_filters, 1, 3, block_strides[i])
                    with scope.name_scope('branch_1'):
                        gamma = _conv_block(x, n_filters, 4, 3, block_strides[i])
                        gamma = Activation('sigmoid', name=scope+'sigmoid')(gamma)

                    # tau = 1 - gamma
                    tau = Lambda(lambda arg: 1 - arg, name=scope+'tau')(gamma)

                    # x = (pi * tau) + (shortcut * gamma)
                    x = Lambda(lambda args: (args[0] * args[1]) + (args[2] * args[3]),
                               name=scope+'out_x')([pi, tau, shortcut, gamma])

            x = Activation('relu', name=scope+'relu_out')(x)
            x = Lambda(lambda x: K.mean(x, axis=(1, 2)), name=scope+'mean_pool')(x)

        return x

    def generator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('generator'):
            n_hidden = 32
            block_factors = range(1, self.nblocks + 1)
            block_strides = [2] * self.nblocks

            if not (self.time_pres_emb or self.use_pose_fae):
                for i in range(2):
                    with scope.name_scope('dense_block_%d' % i):
                        if i > 0:
                            # x = InstanceNormalization(axis=-1, name=scope+'inorm')(x)
                            x = Activation('relu', name=scope+'relu')(x)
                        x = Dense(n_hidden * 4, name=scope+'dense')(x)

                # x = InstanceNormalization(axis=-1, name=scope+'inorm_conv_in')(x)
                x = Activation('relu', name=scope+'relu_conv_in')(x)
                x = Dense(4 * 4 * n_hidden * block_factors[0], name=scope+'dense_conv_in')(x)
                x = Reshape((4, 4, n_hidden * block_factors[0]), name=scope+'reshape_conv_in')(x)

            for i, factor in enumerate(block_factors):
                with scope.name_scope('block_%d' % i):
                    n_filters = n_hidden * factor
                    strides = block_strides[i]
                    if self.time_pres_emb:
                        strides = (block_strides[i], 1)
                    elif self.use_pose_fae:
                        strides = 1
                    shortcut = Conv2DTranspose(n_filters, strides, strides,
                                               name=scope+'shortcut', **CONV2D_ARGS)(x)
                    with scope.name_scope('branch_0'):
                        pi = _conv_block(x, n_filters, 1, 3, strides, Conv2DTranspose)
                    with scope.name_scope('branch_1'):
                        gamma = _conv_block(x, n_filters, 4, 3, strides, Conv2DTranspose)
                        gamma = Activation('sigmoid', name=scope+'gamma_sigmoid')(gamma)

                    # tau = 1 - gamma
                    tau = Lambda(lambda arg: 1 - arg, name=scope+'tau')(gamma)

                    # x = (pi * tau) + (shortcut * gamma)
                    x = Lambda(lambda args: (args[0] * args[1]) + (args[2] * args[3]),
                               name=scope+'out_x')([pi, tau, shortcut, gamma])

            # x = InstanceNormalization(axis=-1, name=scope+'inorm_out')(x)
            x = Activation('relu', name=scope+'relu_out')(x)

        return x


def _preact_dense(x, n_units):
    scope = Scoping.get_global_scope()
    # x = InstanceNormalization(name=scope+'inorm')(x)
    x = Activation('relu', name=scope+'relu')(x)
    x = Dense(n_units, name=scope+'dense', activation='relu')(x)
    return x


class MotionGANV3(_MotionGAN):
    # Simple dense ResNet

    def discriminator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('discriminator'):
            n_hidden = 512

            x = Reshape((self.njoints * self.seq_len * 3, ), name=scope+'reshape_in')(x)
            x = Dense(n_hidden, name=scope+'dense_0', activation='relu')(x)
            for i in range(4):
                with scope.name_scope('block_%d' % i):
                    pi = Dense(n_hidden, name=scope+'dense_0', activation='relu')(x)
                    pi = Dense(n_hidden, name=scope+'dense_1', activation='relu')(pi)

                    x = Add(name=scope+'add')([x, pi])

        return x

    def generator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('generator'):
            n_hidden = 512

            x = Flatten(name=scope+'flatten_in')(x)
            x = _preact_dense(x, n_hidden)
            for i in range(4):
                with scope.name_scope('block_%d' % i):
                    with scope.name_scope('pi_0'):
                        pi = _preact_dense(x, n_hidden)
                    with scope.name_scope('pi_1'):
                        pi = _preact_dense(pi, n_hidden)

                    x = Add(name=scope+'add')([x, pi])

            if self.use_pose_fae:
                fae_dim = self.org_shape[1] * self.org_shape[3] * 2
                x = Dense((self.seq_len * fae_dim),
                          name=scope+'dense_out', activation='relu')(x)
                x = Reshape((self.seq_len, fae_dim, 1),
                            name=scope+'reshape_out')(x)
            else:
                x = Dense((self.njoints * self.seq_len * 3),
                          name=scope+'dense_out', activation='relu')(x)
                x = Reshape((self.njoints, self.seq_len, 3),
                            name=scope+'reshape_out')(x)

        return x


class MotionGANV4(_MotionGAN):
    # ResNet Discriminator, Dilated ResNet + UNet Generator

    def discriminator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('discriminator'):
            n_hidden = 64
            block_factors = [1, 1, 2, 2]
            block_strides = [2, 2, 1, 1]

            x = Conv2D(n_hidden * block_factors[0], 3, 1, name=scope+'conv_in', **CONV2D_ARGS)(x)
            for i, factor in enumerate(block_factors):
                with scope.name_scope('block_%d' % i):
                    n_filters = n_hidden * factor
                    shortcut = Conv2D(n_filters, block_strides[i], block_strides[i],
                                      name=scope+'shortcut', **CONV2D_ARGS)(x)
                    with scope.name_scope('branch_0'): # scope for backward compat
                        pi = _conv_block(x, n_filters, 1, 3, block_strides[i])

                    x = Add(name=scope+'add')([shortcut, pi])

            x = Activation('relu', name=scope+'relu_out')(x)
            x = Lambda(lambda x: K.mean(x, axis=(1, 2)), name=scope+'mean_pool')(x)

        return x

    def generator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('generator'):
            n_hidden = 32
            plain_blocks = 1
            u_blocks = 0
            min_space = min(int(x.shape[1]), int(x.shape[2]))
            while min_space > 2:
                min_space //= 2
                u_blocks += 1
            u_blocks = u_blocks * 2
            block_factors = ([1] * plain_blocks) + range(1, (u_blocks // 2) + 1) + range(u_blocks // 2, 0, -1) + ([1] * plain_blocks)
            block_strides = ([1] * plain_blocks) + ([2] * u_blocks) + ([1] * plain_blocks)

            if not (self.time_pres_emb or self.use_pose_fae):
                x = Dense(4 * 4 * n_hidden * block_factors[0], name=scope+'dense_in')(x)
                x = Reshape((4, 4, n_hidden * block_factors[0]), name=scope+'reshape_in')(x)

            conv_args = CONV2D_ARGS.copy()
            conv_args['padding'] = 'valid'
            u_skips = []
            for i, factor in enumerate(block_factors):
                with scope.name_scope('block_%d' % i):
                    n_filters = n_hidden * factor
                    strides = block_strides[i]
                    # if self.time_pres_emb:
                    #     strides = (block_strides[i], 1)
                    # elif self.use_pose_fae:
                    #     strides = 1
                    if i < (u_blocks // 2) + plain_blocks:
                        conv_func = Conv2D
                        if i > plain_blocks - 1:
                            u_skips.append(x)
                    else:
                        conv_func = Conv2DTranspose

                    pis = []
                    for j in range(2):
                        with scope.name_scope('branch_%d' % j):
                            pis.append(_conv_block(x, n_filters, 1, 3, 1,
                                                   conv_func, (1, 2 ** j)))

                    pi = Concatenate(name=scope+'cat_pi_0')(pis)
                    pi = Activation('relu', name=scope+'relu_pi_0')(pi)
                    pi = conv_func(n_filters, strides, strides, name=scope+'reduce_pi_0', **conv_args)(pi)

                    if (u_blocks // 2) + plain_blocks <= i < u_blocks + plain_blocks:
                        skip_pi = u_skips.pop()
                        if skip_pi.shape[1] != pi.shape[1] or skip_pi.shape[2] != pi.shape[2]:
                            pi = ZeroPadding2D(((0, int(skip_pi.shape[1] - pi.shape[1])),
                                                (0, int(skip_pi.shape[2] - pi.shape[2]))),
                                               name=scope+'pad_pi')(pi)
                        pi = Concatenate(name=scope+'cat_pi_1')([skip_pi, pi])
                        pi = Activation('relu', name=scope+'relu_pi_1')(pi)
                        pi = conv_func(n_filters, 3, 1, name=scope+'reduce_pi_1', **CONV2D_ARGS)(pi)

                    if i < u_blocks - 1:
                        shortcut = conv_func(n_filters, strides, strides, name=scope+'shortcut', **conv_args)(x)
                        if pi.shape[1] != shortcut.shape[1] or pi.shape[2] != shortcut.shape[2]:
                            shortcut = ZeroPadding2D(((0, int(pi.shape[1] - shortcut.shape[1])),
                                                      (0, int(pi.shape[2] - shortcut.shape[2]))),
                                                     name=scope+'pad_short')(shortcut)
                        x = Add(name=scope+'add')([shortcut, pi])
                    else:
                        x = pi

            # x = InstanceNormalization(axis=-1, name=scope+'inorm_out')(x)
            # x = Activation('relu', name=scope+'relu_out')(x)

        return x


class MotionGANV5(_MotionGAN):
    # ResNet, WaveNet style generator

    def discriminator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('discriminator'):
            n_hidden = 64
            block_factors = [1, 1, 2, 2]
            block_strides = [2, 2, 1, 1]

            x = Conv2D(n_hidden * block_factors[0], 3, 1, name=scope+'conv_in', **CONV2D_ARGS)(x)
            for i, factor in enumerate(block_factors):
                with scope.name_scope('block_%d' % i):
                    n_filters = n_hidden * factor
                    shortcut = Conv2D(n_filters, block_strides[i], block_strides[i],
                                      name=scope+'shortcut', **CONV2D_ARGS)(x)
                    with scope.name_scope('branch_0'): # scope for backward compat
                        pi = _conv_block(x, n_filters, 1, 3, block_strides[i])

                    x = Add(name=scope+'add')([shortcut, pi])

            x = Activation('relu', name=scope+'relu_out')(x)
            x = Lambda(lambda x: K.mean(x, axis=(1, 2)), name=scope+'mean_pool')(x)

        return x

    def generator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('generator'):
            x_shape = [int(dim) for dim in x.shape]
            n_hidden = 64
            time_steps = x_shape[1]
            n_blocks = 0
            while time_steps > 1:
                time_steps //= 2
                n_blocks += 1

            if not self.use_pose_fae:
                x = Permute((2, 1, 3), name=scope+'perm_in')(x)

            with scope.name_scope('wave_gen'):
                wave_input = Input(batch_shape=(x_shape[0], x_shape[1] // 2, x_shape[2], x_shape[3]))
                # print(time_steps, n_blocks)

                wave_output = wave_input
                for i in range(n_blocks):
                    with scope.name_scope('block_%d' % i):
                        n_filters = n_hidden * (i + 1)
                        pi = _conv_block(wave_output, n_filters, 2, 3, (2, 1), Conv2D)
                        shortcut = Conv2D(n_filters, (2, 1), (2, 1), name=scope+'shortcut', **CONV2D_ARGS)(wave_output)
                        wave_output = Add(name=scope+'add')([shortcut, pi])

                wave_output = Conv2D(1, 1, 1, name=scope+'merge_out', **CONV2D_ARGS)(wave_output)
                wave_output = Reshape((x_shape[2], 1), name=scope+'squeeze_out')(wave_output)

                wave_gen = Model(wave_input, wave_output, name='wave_gen_model')

            # print(wave_gen.summary())

            xs = []
            for i in range(x_shape[1] // 2):
                with scope.name_scope('wave_gen_call_%d' % i):
                    x_step = Lambda(lambda arg: K.stop_gradient(arg[:, i:x_shape[1] // 2, ...]),
                                    name=scope+'wave_in_slice')(x)
                    if i > 0:
                        if len(xs) > 1:
                            x_step_n = Lambda(lambda arg: K.stack(arg, axis=1),
                                              name=scope+'wave_stack_n')(xs)
                        else:
                            x_step_n = Lambda(lambda arg: K.expand_dims(arg, axis=1),
                                              name=scope+'wave_stack_n')(xs[0])
                        x_step = Lambda(lambda arg: K.concatenate(arg, axis=1),
                                        name=scope+'wave_append_n')([x_step, x_step_n])
                    pred_x = wave_gen(x_step)
                    pred_x = Lambda(lambda args: K.concatenate([args[1], args[0][:, 0, :, 1:]], axis=-1),
                                    name=scope+'cat_label_pred')([x_step, pred_x])
                    xs.append(pred_x)

            x = Lambda(lambda arg: arg[:, :x_shape[1] // 2, :, :], name=scope+'slice_out')(x)
            x = Reshape((x_shape[1] // 2, x_shape[2], x_shape[3]), name=scope+'res_slice_out')(x)
            xs = Lambda(lambda arg: K.stack(arg, axis=1), name=scope+'stack_out')(xs)
            xs = Reshape((x_shape[1] // 2, x_shape[2], x_shape[3]), name=scope+'res_stack_out')(xs)
            x = Concatenate(axis=1, name=scope+'cat_out')([x, xs])
            x = Lambda(lambda arg: arg[:, :, :, 0], name=scope+'trim_out')(x)
            x = Reshape((x_shape[1], x_shape[2], 1), name=scope+'res_trim_out')(x)

            if not self.use_pose_fae:
                x = Permute((2, 1, 3), name=scope+'perm_out')(x)

        return x


class MotionGANV6(_MotionGAN):
    # ResNet Discriminator, Dilated ResNet + UNet Generator 4 STACK

    def discriminator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('discriminator'):
            n_hidden = 64
            block_factors = [1, 1, 2, 2]
            block_strides = [2, 2, 1, 1]

            x = Conv2D(n_hidden * block_factors[0], 3, 1, name=scope+'conv_in', **CONV2D_ARGS)(x)
            for i, factor in enumerate(block_factors):
                with scope.name_scope('block_%d' % i):
                    n_filters = n_hidden * factor
                    shortcut = Conv2D(n_filters, block_strides[i], block_strides[i],
                                      name=scope+'shortcut', **CONV2D_ARGS)(x)
                    with scope.name_scope('branch_0'):  # scope for backward compat
                        pi = _conv_block(x, n_filters, 1, 3, block_strides[i])

                    x = Add(name=scope+'add')([shortcut, pi])

            x = Activation('relu', name=scope+'relu_out')(x)
            x = Lambda(lambda x: K.mean(x, axis=(1, 2)), name=scope+'mean_pool')(x)

        return x

    def generator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('generator'):
            n_hidden = 32
            u_blocks = 0
            min_space = min(int(x.shape[1]), int(x.shape[2]))
            while min_space > 2:
                min_space //= 2
                u_blocks += 1
            u_blocks = u_blocks * 2
            block_factors = range(1, (u_blocks // 2) + 1) + range(u_blocks // 2, 0, -1)
            block_strides = ([2] * u_blocks)
            macro_blocks = 4

            if not (self.time_pres_emb or self.use_pose_fae):
                x = Dense(4 * 4 * n_hidden * block_factors[0], name=scope+'dense_in')(x)
                x = Reshape((4, 4, n_hidden * block_factors[0]), name=scope+'reshape_in')(x)

            conv_args = CONV2D_ARGS.copy()
            conv_args['padding'] = 'valid'
            u_skips = []
            for k in range(macro_blocks):
                with scope.name_scope('macro_block_%d' % k):
                    if k < macro_blocks - 1:
                        macro_shortcut = Conv2D(n_hidden, 1, 1, name=scope+'shortcut', **conv_args)(x)
                    for i, factor in enumerate(block_factors):
                        with scope.name_scope('block_%d' % i):
                            n_filters = n_hidden * factor
                            strides = block_strides[i]
                            if i < (u_blocks // 2):
                                conv_func = Conv2D
                                u_skips.append(x)
                            else:
                                conv_func = Conv2DTranspose

                            pi = conv_func(n_filters, strides, strides, name=scope+'reduce_pi_0', **conv_args)(x)

                            if (u_blocks // 2) <= i < u_blocks:
                                skip_pi = u_skips.pop()
                                if skip_pi.shape[1] != pi.shape[1] or skip_pi.shape[2] != pi.shape[2]:
                                    pi = ZeroPadding2D(((0, int(skip_pi.shape[1] - pi.shape[1])),
                                                        (0, int(skip_pi.shape[2] - pi.shape[2]))),
                                                       name=scope+'pad_pi')(pi)
                                pi = Concatenate(name=scope+'cat_pi_1')([skip_pi, pi])
                                pi = Activation('relu', name=scope+'relu_pi_1')(pi)
                                pi = conv_func(n_filters, 3, 1, name=scope+'reduce_pi_1', **CONV2D_ARGS)(pi)

                            shortcut = conv_func(n_filters, strides, strides, name=scope+'shortcut', **conv_args)(x)
                            if pi.shape[1] != shortcut.shape[1] or pi.shape[2] != shortcut.shape[2]:
                                shortcut = ZeroPadding2D(((0, int(pi.shape[1] - shortcut.shape[1])),
                                                          (0, int(pi.shape[2] - shortcut.shape[2]))),
                                                         name=scope+'pad_short')(shortcut)
                            x = Add(name=scope+'add_short')([shortcut, pi])

                    if k < macro_blocks - 1:
                        x = Concatenate(axis=-1, name=scope+'cat_short')([macro_shortcut, x])
        return x


class MotionGANV7(_MotionGAN):
    # DMNN Discriminator, Dilated ResNet + UNet Generator 4 STACK

    def discriminator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('discriminator'):

            def resnet_disc(x):
                with scope.name_scope('resnet'):
                    n_hidden = 64
                    block_factors = [1, 2, 4]
                    block_strides = [2, 2, 2]

                    x = Conv2D(n_hidden * block_factors[0], 3, 1, name=scope+'conv_in', **CONV2D_ARGS)(x)
                    for i, factor in enumerate(block_factors):
                        with scope.name_scope('block_%d' % i):
                            n_filters = n_hidden * factor
                            shortcut = Conv2D(n_filters, block_strides[i], block_strides[i],
                                              name=scope+'shortcut', **CONV2D_ARGS)(x)
                            pi = _conv_block(x, n_filters, 2, 3, block_strides[i])
                            x = Add(name=scope+'add')([shortcut, pi])

                    x = Activation('relu', name=scope+'relu_out')(x)
                    x = Lambda(lambda x: K.mean(x, axis=(1, 2)), name=scope+'mean_pool')(x)
                return x

            def dmnn_disc(x):
                with scope.name_scope('dmnn'):
                    blocks = [{'size': 64,  'bneck_f': 4, 'strides': 2},
                              {'size': 128, 'bneck_f': 4, 'strides': 2},
                              {'size': 256, 'bneck_f': 4, 'strides': 2}]
                    n_reps = 2

                    x = CombMatrix(self.njoints, name=scope+'comb_matrix')(x)
        
                    x = EDM(name=scope+'edms')(x)
                    x = Reshape((self.njoints * self.njoints, self.seq_len, 1), name=scope+'resh_in')(x)
        
                    # x = InstanceNormalization(axis=-1, name=scope+'inorm_in')(x)
                    x = Conv2D(blocks[0]['size'] // blocks[0]['bneck_f'], 1, 1, name=scope+'conv_in', **CONV2D_ARGS)(x)
                    for i in range(len(blocks)):
                        for j in range(n_reps):
                            with scope.name_scope('block_%d_%d' % (i, j)):
                                strides = blocks[i]['strides'] if j == 0 else 1
                                if int(x.shape[-1]) != blocks[i]['size'] or strides > 1:
                                    with scope.name_scope('shortcut'):
                                        shortcut = Activation('relu', name=scope+'relu')(x)
                                        shortcut = Conv2D(blocks[i]['size'], 1, strides,
                                                          name=scope+'conv', **CONV2D_ARGS)(shortcut)
                                else:
                                    shortcut = x
        
                                x = _conv_block(x, blocks[i]['size'], blocks[i]['bneck_f'], 3, strides)
                                x = Add(name=scope+'add')([shortcut, x])
        
                    x = Lambda(lambda args: K.mean(args, axis=(1, 2)), name=scope+'mean_pool')(x)
                return x

            x = Concatenate(axis=-1, name=scope+'features_cat')([resnet_disc(x), dmnn_disc(x)])

        return x

    def generator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('generator'):
            n_hidden = 32
            u_blocks = 0
            min_space = min(int(x.shape[1]), int(x.shape[2]))
            while min_space > 2:
                min_space //= 2
                u_blocks += 1
            u_blocks = u_blocks * 2
            block_factors = range(1, (u_blocks // 2) + 1) + range(u_blocks // 2, 0, -1)
            block_strides = ([2] * u_blocks)
            macro_blocks = 4

            if not (self.time_pres_emb or self.use_pose_fae):
                x = Dense(4 * 4 * n_hidden * block_factors[0], name=scope+'dense_in')(x)
                x = Reshape((4, 4, n_hidden * block_factors[0]), name=scope+'reshape_in')(x)

            u_skips = []
            x = Conv2D(n_hidden, 1, 1, name=scope+'conv_in', **CONV2D_ARGS)(x)
            batch_size = self.batch_size
            for k in range(macro_blocks):
                with scope.name_scope('macro_block_%d' % k):
                    shortcut = x
                    for i, factor in enumerate(block_factors):
                        with scope.name_scope('block_%d' % i):
                            n_filters = n_hidden * factor
                            strides = block_strides[i]
                            if i < (u_blocks // 2):
                                conv_func = Conv2D
                                u_skips.append(x)
                            else:
                                conv_func = Conv2DTranspose

                            with scope.name_scope('pi'):
                                x = _conv_block(x, n_filters, 2, 3, strides, conv_func)

                            if (u_blocks // 2) <= i < u_blocks:
                                skip_x = u_skips.pop()
                                if skip_x.shape[1] != x.shape[1] or skip_x.shape[2] != x.shape[2]:
                                    x = Cropping2D(((0, int(x.shape[1] - skip_x.shape[1])),
                                                    (0, int(x.shape[2] - skip_x.shape[2]))),
                                                     name=scope+'crop_x')(x)
                                x = Concatenate(name=scope+'cat_skip')([skip_x, x])
                                with scope.name_scope('skip_pi'):
                                    x = _conv_block(x, n_filters, 2, 3, 1, conv_func)

                    x = Add(name=scope+'add_short')([shortcut, x])
        return x
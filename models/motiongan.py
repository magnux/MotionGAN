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
    Conv1D, Multiply
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.regularizers import l2
from layers.normalization import InstanceNormalization
from layers.edm import edm, EDM
from layers.comb_matrix import CombMatrix
from collections import OrderedDict
from utils.scoping import Scoping
from utils.tfangles import quaternion_between, quat_to_expmap, expmap_to_rotmat, rotmat_to_euler, \
    vector3d_to_quaternion, quaternion_conjugate, rotate_vector_by_quaternion

CONV1D_ARGS = {'padding': 'same', 'kernel_regularizer': l2(5e-4)}
CONV2D_ARGS = {'padding': 'same', 'data_format': 'channels_last', 'kernel_regularizer': l2(5e-4)}


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
        self.wgan_scale_d = 1.0
        self.wgan_scale_g = 0.1 * (0.0 if config.no_gan_loss else 1.0)
        self.wgan_frame_scale_d = 1.0
        self.wgan_frame_scale_g = 0.1 * (0.0 if config.no_gan_loss else 1.0)
        self.rec_scale = 1.0e-2
        self.action_cond = config.action_cond
        self.action_scale_d = 1.0
        self.action_scale_g = 1.0
        self.latent_cond_dim = config.latent_cond_dim
        self.latent_scale_d = 1.0
        self.latent_scale_g = 1.0
        self.shape_loss = config.shape_loss
        self.shape_scale = 1.0e-2
        self.smoothing_loss = config.smoothing_loss
        self.smoothing_scale = 1.0
        self.smoothing_basis = 5
        self.time_pres_emb = config.time_pres_emb
        self.use_pose_fae = config.use_pose_fae
        self.fae_original_dim = self.njoints * 3
        self.fae_intermediate_dim = self.fae_original_dim
        self.fae_latent_dim = self.fae_original_dim // 2
        self.rotation_loss = config.rotation_loss
        self.rotation_scale = 1.0
        self.translate_start = config.translate_start
        self.rotate_start = config.rotate_start
        self.rescale_coords = config.rescale_coords
        self.remove_hip = config.remove_hip
        self.use_diff = config.use_diff
        self.diff_scale = 1.0e1
        self.use_angles = config.use_angles
        self.angles_scale = 1.0
        self.stats = {}

        # Placeholders for training phase
        self.place_holders = []
        if self.action_cond:
            true_label = K.placeholder(shape=(self.batch_size,), dtype='int32', name='true_label')
            self.place_holders.append(true_label)

        # Discriminator
        real_seq = Input(batch_shape=(self.batch_size, self.njoints, self.seq_len, 3),
                         name='real_seq', dtype='float32')
        self.disc_inputs = [real_seq]
        x = self._proc_disc_inputs(self.disc_inputs)
        self.real_outputs = self._proc_disc_outputs(self.discriminator(x))
        self.disc_model = Model(self.disc_inputs,
                                self.real_outputs,
                                name=self.name + '_discriminator')

        # Generator
        seq_mask = Input(batch_shape=(self.batch_size, self.njoints, self.seq_len, 1),
                         name='seq_mask', dtype='float32')
        self.gen_inputs = [real_seq, seq_mask]
        if self.latent_cond_dim > 0:
            latent_cond_input = Input(batch_shape=(self.batch_size, self.latent_cond_dim),
                                      name='latent_cond_input', dtype='float32')
            self.gen_inputs.append(latent_cond_input)
        x = self._proc_gen_inputs(self.gen_inputs)
        self.gen_outputs = self._proc_gen_outputs(self.generator(x))
        self.gen_model = Model(self.gen_inputs,
                               self.gen_outputs,
                               name=self.name + '_generator')
        self.fake_outputs = self.disc_model(self.gen_outputs[0])

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
            disc_optimizer = Adam(lr=config.learning_rate, beta_1=0., beta_2=0.9)
            disc_training_updates = disc_optimizer.get_updates(disc_loss, self.disc_model.trainable_weights)
            self.disc_train_f = K.function(self.disc_inputs + self.gen_inputs + self.place_holders,
                                           self.wgan_losses.values() + self.disc_losses.values(),
                                           disc_training_updates)

        with K.name_scope('discriminator/functions/eval'):
            self.disc_eval_f = K.function(self.disc_inputs + self.gen_inputs + self.place_holders,
                                          self.wgan_losses.values() + self.disc_losses.values())

        self.disc_model = self._pseudo_build_model(self.disc_model, disc_optimizer)

        with K.name_scope('generator/functions/train'):
            gen_optimizer = Adam(lr=config.learning_rate, beta_1=0., beta_2=0.9)
            gen_training_updates = gen_optimizer.get_updates(gen_loss,
                                   self.gen_model.trainable_weights)
            self.gen_train_f = K.function(self.gen_inputs + self.place_holders,
                                          self.gen_losses.values(),
                                          gen_training_updates)

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

                frame_inter_score = _get_tensor(inter_outputs, 'frame_score_out')
                frame_grad_mixed = K.gradients(frame_inter_score, [interpolates])[0]
                frame_norm_grad_mixed = K.sqrt(K.sum(K.sum(K.square(frame_grad_mixed), axis=(1, 3)) * no_zero_frames, axis=1) + K.epsilon())
                frame_grad_penalty = K.mean(K.square(frame_norm_grad_mixed - self.gamma_grads) / (self.gamma_grads ** 2), axis=-1)

                # WGAN-GP losses
                frame_disc_loss_wgan = frame_loss_fake - frame_loss_real + (self.lambda_grads * frame_grad_penalty)
                disc_losses['frame_disc_loss_wgan'] = self.wgan_frame_scale_d * K.mean(frame_disc_loss_wgan)

                frame_gen_loss_wgan = -frame_loss_fake
                gen_losses['frame_gen_loss_wgan'] = self.wgan_frame_scale_g * K.mean(frame_gen_loss_wgan)

            # Reconstruction loss
            with K.name_scope('reconstruction_loss'):
                loss_rec = K.sum(K.mean(K.square((real_seq * seq_mask) - (gen_seq * seq_mask)), axis=-1), axis=(1, 2))
                gen_losses['gen_loss_rec'] = self.rec_scale * K.mean(loss_rec)

                if self.use_diff:
                    loss_rec_diff = K.sum(K.mean(K.square((self.diff_input * self.diff_input_mask) -
                                                          (self.diff_output * self.diff_input_mask)), axis=-1), axis=(1, 2))
                    gen_losses['gen_loss_rec_diff'] = self.diff_scale * K.mean(loss_rec_diff)

            # Optional losses
            if self.action_cond:
                with K.name_scope('action_loss'):
                    loss_class_real = K.mean(K.sparse_categorical_crossentropy(
                        _get_tensor(self.place_holders, 'true_label'),
                        _get_tensor(self.real_outputs, 'label_out'), True))
                    loss_class_fake = K.mean(K.sparse_categorical_crossentropy(
                        _get_tensor(self.place_holders, 'true_label'),
                        _get_tensor(self.fake_outputs, 'label_out'), True))
                    disc_losses['disc_loss_action'] = self.action_scale_d * (loss_class_real + loss_class_fake)
                    gen_losses['gen_loss_action'] = self.action_scale_g * loss_class_fake
            if self.latent_cond_dim > 0:
                with K.name_scope('latent_loss'):
                    loss_latent = K.mean(K.square(_get_tensor(self.fake_outputs, 'latent_cond_out')
                                                  - _get_tensor(self.gen_inputs, 'latent_cond_input')))
                    disc_losses['disc_loss_latent'] = self.latent_scale_d * loss_latent
                    gen_losses['gen_loss_latent'] = self.latent_scale_g * loss_latent
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
                    loss_rot = K.square(1 - K.batch_dot(unit_real, unit_gen, axes=1))
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

    def _remove_hip_in(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('remove_hip'):

            def _get_hips(arg):
                return K.reshape(arg[:, 0, :, :], (arg.shape[0], 1, arg.shape[2], 3))

            self.stats[scope+'hip_coords'] = Lambda(_get_hips, name=scope+'hip_coords')(x)

            x = Lambda(lambda args: args[0] - args[1], name=scope+'remove_hip_in')([x, self.stats[scope+'hip_coords']])
        return x

    def _remove_hip_out(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('remove_hip'):

            x = Lambda(lambda args: args[0] + args[1], name=scope+'remove_hip_out')([x, self.stats[scope+'hip_coords']])
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
                torso_rot = tf.cross(arg[:, left_shoulder, 0, :] - arg[:, hip, 0, :],
                                     arg[:, right_shoulder, 0, :] - arg[:, hip, 0, :])
                side_rot = K.reshape(tf.cross(arg[:, head_top, 0, :] - arg[:, hip, 0, :], torso_rot), base_shape)
                theta_diff = ((np.pi / 2) - tf.atan2(side_rot[..., 1], side_rot[..., 0])) / 2
                cos_theta_diff = tf.cos(theta_diff)
                sin_theta_diff = tf.sin(theta_diff)
                zeros_theta = K.zeros_like(sin_theta_diff)
                return tf.stack([cos_theta_diff, zeros_theta, zeros_theta, sin_theta_diff], axis=-1)

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

            members_from, members_to, body_graph = self._get_body_graph()

            def _len(bone):
                return K.sqrt(K.sum(K.square(bone), axis=-1, keepdims=True) + K.epsilon())

            def _get_avg_bone_len(arg):
                bones = [arg[:, i, 0, :] - arg[:, j, 0, :] for i, j in zip(members_from, members_to)]
                bones = K.expand_dims(K.stack(bones, axis=1), axis=2)
                return K.mean(_len(bones), axis=1, keepdims=True)

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
                x_mask = Lambda(lambda arg: arg[:, :, 1:, :] * arg[:, :, :-1, :], name=scope + 'seq_mask_to_diff_in')(x_mask)
        return x, x_mask

    def _seq_to_diff_out(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('seq_to_diff'):

            def _diff_to_seq(args):
                diffs, start_pose = args
                poses = [start_pose]
                for p in range(diffs.shape[2]):
                    poses.append(poses[p] + diffs[:, :, p, :])
                return tf.stack(poses, axis=2)

            x = Lambda(_diff_to_seq, name=scope+'seq_to_diff_out')([x, self.stats[scope+'start_pose']])
        return x

    def _get_body_graph(self):
        members_from = []
        members_to = []
        for member in self.body_members.values():
            for j in range(len(member['joints']) - 1):
                members_from.append(member['joints'][j])
                members_to.append(member['joints'][j + 1])

        members_lst = zip(members_from, members_to)

        graph = {name: set() for tup in members_lst for name in tup}
        has_parent = {name: False for tup in members_lst for name in tup}
        for parent, child in members_lst:
            graph[parent].add(child)
            has_parent[child] = True

        # roots = [name for name, parents in has_parent.items() if not parents]  # assuming 0 (hip)
        #
        # def traverse(hierarchy, graph, names):
        #     for name in names:
        #         hierarchy[name] = traverse({}, graph, graph[name])
        #     return hierarchy
        # traverse({}, graph, roots)

        return members_from, members_to, graph

    def _seq_to_angles_in(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('seq_to_angles'):

            members_from, members_to, body_graph = self._get_body_graph()

            def _get_hips(arg):
                return K.reshape(arg[:, 0, :, :], (arg.shape[0], 1, self.seq_len, 3))
            self.stats[scope+'hip_coords'] = Lambda(_get_hips, name=scope+'hip_coords')(x)

            def _get_bones(arg):
                return arg[:, members_from, 0, :] - arg[:, members_to, 0, :]

            self.stats[scope+'bones'] = Lambda(_get_bones, name=scope+'bones')(x)

            def _len(bone):
                return K.sqrt(K.sum(K.square(bone), axis=-1, keepdims=True) + K.epsilon())

            self.stats[scope+'bone_len'] = Lambda(_len, name=scope+'bone_len')(self.stats[scope+'bones'])

            # TODO: angles are not computed correctly
            def _get_angles(arg):
                angles = []
                def _get_angles(parent_idx, idx):
                    if parent_idx is None:
                        pass  # TODO: add base cases
                    else:
                        angle = quat_to_expmap(quaternion_between(self.stats[scope+'bones'][parent_idx],
                                                                  self.stats[scope+'bones'][idx]))
                        angles.append(angle)
                    for child_idx in body_graph[idx]:
                        _get_angles(idx, child_idx)

                return tf.stack(angles, axis=1)

            x = Lambda(_get_angles, name=scope+'angles')(x)
        return x

    def _seq_to_angles_out(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('seq_to_angles'):

            members_from, members_to, body_graph = self._get_body_graph()

            x = Lambda(lambda arg: expmap_to_rotmat(arg), name=scope+'rotmat')(x)
            self.euler_out = Lambda(lambda arg: rotmat_to_euler(arg), name=scope+'euler')(x)

            base_shape = [int(d) for d in x.shape]
            base_shape[1] = 1
            base_shape[-1] = 1

            def _get_coords(args):
                rot_mat, bone_len = args
                coords = range(self.njoints)

                def _set_coords(parent_idx, idx):
                    if parent_idx is None:
                        coords[idx] = K.zeros(base_shape)
                    else:
                        coords[idx] = coords[parent_idx]  # + rotation dot bonelen
                        # this rotation dot parent rotation
                        # TODO: implement forward kinematics once the angles are correctly computed

                    for child_idx in body_graph[idx]:
                        _set_coords(coords[idx], child_idx)

                coords = K.stack(coords, axis=1)
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
            if self.remove_hip:
                x = self._remove_hip_in(x)
            if self.use_diff:
                x, _ = self._seq_to_diff_in(x)

        return x

    def _proc_disc_outputs(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('discriminator'):
            score_out = Dense(1, name=scope+'score_out')(x)

            output_tensors = [score_out]
            if self.action_cond:
                label_out = Dense(self.num_actions, name=scope+'label_out')(x)
                output_tensors.append(label_out)
            if self.latent_cond_dim > 0:
                latent_cond_out = Dense(self.latent_cond_dim, name=scope+'latent_cond_out')(x)
                output_tensors.append(latent_cond_out)

            seq = self.disc_inputs[0]

            z = self._pose_encoder(seq)

            frame_score_out = Conv1D(1, 1, 1, name=scope+'frame_score_out', **CONV1D_ARGS)(z)
            output_tensors.append(frame_score_out)

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
                # self.aux_out = x
                # self.aux_out = self._rotate_start_out(x)
            if self.rescale_coords:
                x = self._rescale_in(x)
            if self.remove_hip:
                x = self._remove_hip_in(x)
            if self.use_diff:
                x, x_mask = self._seq_to_diff_in(x, x_mask)
                self.diff_input, self.diff_input_mask = x, x_mask


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

            x = [x]
            if self.latent_cond_dim > 0:
                x_lat = _get_tensor(input_tensors, 'latent_cond_input')
                x.append(x_lat)

            if len(x) > 1:
                x = Concatenate(name=scope+'cat_in')(x)
            else:
                x = x[0]

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
                x = Conv2D(self.njoints, 3, 1, name=scope+'joint_reshape', **CONV2D_ARGS)(x)
                x = Permute((1, 3, 2), name=scope+'time_permute')(x)  # filters, joints, time
                x = Conv2D(self.seq_len, 3, 1, name=scope+'time_reshape', **CONV2D_ARGS)(x)
                x = Permute((2, 3, 1), name=scope+'coords_permute')(x)  # joints, time, filters
                x = Conv2D(3, 3, 1, name=scope+'coords_reshape', **CONV2D_ARGS)(x)

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

            h = Permute((2, 1, 3), name=scope+'perm_in')(seq)
            h = Reshape((int(seq.shape[2]), int(seq.shape[1] * seq.shape[3])), name=scope+'resh_in')(h)

            h = Conv1D(self.fae_intermediate_dim, 1, 1,
                       name=scope+'conv_in', **CONV1D_ARGS)(h)
            for i in range(3):
                with scope.name_scope('block_%d' % i):
                    pi = Conv1D(self.fae_intermediate_dim, 1, 1, activation='relu',
                                name=scope+'pi_0', **CONV1D_ARGS)(h)
                    pi = Conv1D(self.fae_intermediate_dim, 1, 1, activation='relu',
                                name=scope+'pi_1', **CONV1D_ARGS)(pi)
                    tau = Conv1D(self.fae_intermediate_dim, 1, 1, activation='sigmoid',
                                 name=scope+'tau_0', **CONV1D_ARGS)(h)
                    h = Lambda(lambda args: (args[0] * (1 - args[2])) + (args[1] * args[2]),
                               name=scope+'attention')([h, pi, tau])

            z = Conv1D(self.fae_latent_dim, 1, 1, name=scope+'z_mean', **CONV1D_ARGS)(h)
            z_attention = Conv1D(self.fae_latent_dim, 1, 1, activation='sigmoid',
                                 name=scope+'attention_mask', **CONV1D_ARGS)(h)

            # We are only expecting half of the latent features to be activated
            z = Multiply(name=scope+'z_attention')([z, z_attention])

        return z

    def _pose_decoder(self, gen_z):
        scope = Scoping.get_global_scope()
        with scope.name_scope('decoder'):

            dec_h = Conv1D(self.fae_intermediate_dim, 1, 1,
                           name=scope+'conv_in', **CONV1D_ARGS)(gen_z)
            for i in range(3):
                with scope.name_scope('block_%d' % i):
                    pi = Conv1D(self.fae_intermediate_dim, 1, 1, activation='relu',
                                name=scope+'pi_0', **CONV1D_ARGS)(dec_h)
                    pi = Conv1D(self.fae_intermediate_dim, 1, 1, activation='relu',
                                name=scope+'pi_1', **CONV1D_ARGS)(pi)
                    tau = Conv1D(self.fae_intermediate_dim, 1, 1, activation='sigmoid',
                                 name=scope+'tau_0', **CONV1D_ARGS)(dec_h)
                    dec_h = Lambda(lambda args: (args[0] * (1 - args[2])) + (args[1] * args[2]),
                                   name=scope+'attention')([dec_h, pi, tau])

            dec_x = Conv1D(self.fae_original_dim, 1, 1, name=scope+'conv_out', **CONV1D_ARGS)(dec_h)

            dec_x = Reshape((int(gen_z.shape[1]), self.njoints, 3), name=scope+'resh_out')(dec_x)
            dec_x = Permute((2, 1, 3), name=scope+'perm_out')(dec_x)

        return dec_x


def _conv_block(x, out_filters, bneck_factor, kernel_size, strides, conv_func=Conv2D, dilation_rate=(1, 1)):
    scope = Scoping.get_global_scope()
    if 'generator' in str(scope):
        x = InstanceNormalization(axis=-1, name=scope+'inorm_in')(x)
    x = Activation('relu', name=scope+'relu_in')(x)
    x = conv_func(filters=out_filters // bneck_factor,
                  kernel_size=kernel_size, strides=1,
                  dilation_rate=dilation_rate, name=scope+'conv_in', **CONV2D_ARGS)(x)
    if 'generator' in str(scope):
        x = InstanceNormalization(axis=-1, name=scope+'inorm_out')(x)
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
                    shortcut = Conv2DTranspose(n_filters, strides, strides,
                                               name=scope+'shortcut', **CONV2D_ARGS)(x)
                    with scope.name_scope('branch_0'): # scope for backward compat
                        pi = _conv_block(x, n_filters, 1, 3, strides, Conv2DTranspose)

                    x = Add(name=scope+'add')([shortcut, pi])

            x = InstanceNormalization(axis=-1, name=scope+'inorm_out')(x)
            x = Activation('relu', name=scope+'relu_out')(x)

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
                            x = InstanceNormalization(axis=-1, name=scope+'bn')(x)
                            x = Activation('relu', name=scope+'relu')(x)
                        x = Dense(n_hidden * 4, name=scope+'dense')(x)

                x = InstanceNormalization(axis=-1, name=scope+'inorm_conv_in')(x)
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

            x = InstanceNormalization(axis=-1, name=scope+'inorm_out')(x)
            x = Activation('relu', name=scope+'relu_out')(x)

        return x


def _preact_dense(x, n_units):
    scope = Scoping.get_global_scope()
    x = InstanceNormalization(name=scope+'inorm')(x)
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
                x = Dense((self.seq_len * self.fae_latent_dim),
                          name=scope+'dense_out', activation='relu')(x)
                x = Reshape((self.seq_len, self.fae_latent_dim, 1),
                            name=scope+'reshape_out')(x)
            else:
                x = Dense((self.njoints * self.seq_len * 3),
                          name=scope+'dense_out', activation='relu')(x)
                x = Reshape((self.njoints, self.seq_len, 3),
                            name=scope+'reshape_out')(x)

        return x


class MotionGANV4(_MotionGAN):
    # DMNN Discriminator, Dilated ResNet Generator

    def discriminator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('classifier'):
            blocks = [{'size': 64,  'bneck_f': 2, 'strides': 3},
                      {'size': 128, 'bneck_f': 2, 'strides': 3}]
            n_reps = 3

            x = CombMatrix(self.njoints, name=scope+'comb_matrix')(x)

            x = EDM(name=scope+'edms')(x)
            x = Reshape((self.njoints * self.njoints, self.seq_len, 1), name=scope+'resh_in')(x)

            x = InstanceNormalization(axis=-1, name=scope+'in_in')(x)
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
            x = InstanceNormalization(name=scope+'in_out')(x)
            x = Activation('relu', name=scope+'relu_out')(x)

            x = Dense(self.num_actions, activation='softmax', name=scope+'label')(x)

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
                    shortcut = Conv2DTranspose(n_filters, strides, strides,
                                               name=scope+'shortcut', **CONV2D_ARGS)(x)

                    pis = []
                    for j in range(4):
                        with scope.name_scope('branch_%d' % j):
                            pis.append(_conv_block(x, n_filters // 4, 1, 3, 1,
                                                   Conv2DTranspose, (1, 2 ** j)))

                    pi = Concatenate(name=scope+'cat_pi')(pis)
                    pi = Conv2DTranspose(n_filters, strides, strides,
                                         name=scope+'reduce_pi', **CONV2D_ARGS)(pi)

                    x = Add(name=scope+'add')([shortcut, pi])

            x = InstanceNormalization(axis=-1, name=scope+'inorm_out')(x)
            x = Activation('relu', name=scope+'relu_out')(x)

        return x

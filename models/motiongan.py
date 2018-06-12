from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow.contrib.keras.api.keras.backend as K
from scipy.fftpack import idct
from scipy.linalg import pinv
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Input
from tensorflow.contrib.keras.api.keras.layers import Conv2DTranspose, Conv2D, \
    Dense, Activation, Lambda, Add, Concatenate, Permute, Reshape, Flatten, \
    Conv1D, Multiply, Cropping2D, Conv3D
from tensorflow.contrib.keras.api.keras.optimizers import Nadam
from tensorflow.contrib.keras.api.keras.regularizers import l2
from layers.edm import edm, EDM
from layers.comb_matrix import CombMatrix
from layers.tile import Tile
from layers.seq_transform import remove_hip_in, remove_hip_out, translate_start_in, translate_start_out, \
    rotate_start_in, rotate_start_out, rescale_body_in, rescale_body_out, \
    seq_to_diff_in, seq_to_diff_out, seq_to_angles_in, seq_to_angles_out
from collections import OrderedDict
from utils.scoping import Scoping

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
        self.rec_scale = 1.0   # if 'expmaps' not in self.data_set else 10.0
        self.action_cond = config.action_cond
        self.action_scale_d = 10.0
        self.action_scale_g = 1.0
        self.coherence_loss = config.coherence_loss
        self.coherence_scale = 1.0
        self.shape_loss = config.shape_loss
        self.shape_scale = 1.0
        self.smoothing_loss = config.smoothing_loss
        self.smoothing_scale = 20.0
        self.smoothing_basis = 5
        self.time_pres_emb = config.time_pres_emb
        self.use_pose_fae = config.use_pose_fae
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
        # self.place_holders = [true_label]  # it is not an input because it is only used in the loss
        x = self._proc_disc_inputs(self.disc_inputs)
        self.real_outputs = self._proc_disc_outputs(self.discriminator(x))
        self.disc_model = Model(self.disc_inputs, self.real_outputs, name=self.name + '_discriminator')

        # Generator
        seq_mask = Input(batch_shape=(self.batch_size, self.njoints, self.seq_len, 1), name='seq_mask', dtype='float32')
        self.gen_inputs = [real_seq, seq_mask]
        if self.action_cond:
            self.gen_inputs.append(true_label)
        x = self._proc_gen_inputs(self.gen_inputs)
        self.gen_outputs = self._proc_gen_outputs(self.generator(x))
        self.gen_model = Model(self.gen_inputs, self.gen_outputs, name=self.name + '_generator')
        self.fake_outputs = self.disc_model(self.gen_outputs)

        # Losses
        self.wgan_losses, self.disc_losses, self.gen_losses, self.gen_metrics = self._build_loss()

        with K.name_scope('loss/sum'):
            disc_loss = 0.0
            for loss in self.disc_losses.values():
                disc_loss += loss

            gen_loss = 0.0
            for loss in self.gen_losses.values():
                gen_loss += loss

        # Custom train functions
        with K.name_scope('discriminator/functions/train'):
            disc_optimizer = Nadam(lr=config.learning_rate)
            disc_training_updates = disc_optimizer.get_updates(disc_loss, self.disc_model.trainable_weights)
            self.disc_train_f = K.function(self.disc_inputs + self.gen_inputs,
                                           self.wgan_losses.values() + self.disc_losses.values(),
                                           disc_training_updates)

        with K.name_scope('discriminator/functions/eval'):
            self.disc_eval_f = K.function(self.disc_inputs + self.gen_inputs,
                                          self.wgan_losses.values() + self.disc_losses.values())

        self.disc_model = self._pseudo_build_model(self.disc_model, disc_optimizer)

        with K.name_scope('generator/functions/train'):
            gen_optimizer = Nadam(lr=config.learning_rate)
            gen_training_updates = gen_optimizer.get_updates(gen_loss, self.gen_model.trainable_weights)
            self.gen_train_f = K.function(self.gen_inputs,
                                          self.gen_losses.values() + self.gen_metrics.values(),
                                          gen_training_updates)

        with K.name_scope('generator/functions/eval'):
            gen_f_outs = self.gen_losses.values() + self.gen_metrics.values()
            if self.use_pose_fae:
                gen_f_outs.append(self.fae_z)
            # gen_f_outs.append(self.aux_out)
            gen_f_outs += self.gen_outputs
            self.gen_eval_f = K.function(self.gen_inputs,  gen_f_outs)

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
        keys = self.gen_losses.keys() + self.gen_metrics.keys()
        keys = ['train/%s' % key for key in keys]
        losses_dict = OrderedDict(zip(keys, train_outs))
        return losses_dict

    def gen_eval(self, inputs):
        eval_outs = self.gen_eval_f(inputs)
        keys = self.gen_losses.keys() + self.gen_metrics.keys()
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
            gen_metrics = OrderedDict()

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

            # Reconstruction loss
            with K.name_scope('reconstruction_loss'):
                loss_rec = K.sum(K.mean(K.square((real_seq - gen_seq) * seq_mask), axis=-1), axis=(1, 2))
                gen_losses['gen_loss_rec'] = self.rec_scale * K.mean(loss_rec)

                if self.use_diff:
                    loss_rec_diff = K.sum(K.mean(K.square((self.diff_input * self.diff_mask) -
                                                          (self.diff_output * self.diff_mask)), axis=-1), axis=(1, 2))
                    gen_losses['gen_loss_rec_diff'] = self.diff_scale * K.mean(loss_rec_diff)
                if self.use_angles:
                    loss_rec_angles = K.sum(K.mean(K.square((self.angles_input * self.angles_mask) -
                                                            (self.angles_output * self.angles_mask)), axis=-1), axis=(1, 2))
                    gen_losses['gen_loss_rec_angles'] = self.angles_scale * K.mean(loss_rec_angles)

                loss_rec = K.sum(K.mean(K.square((real_seq - gen_seq) * (1 - seq_mask)), axis=-1), axis=(1, 2))
                gen_metrics['gen_loss_rec_comp'] = self.rec_scale * K.mean(loss_rec)

            # Action label loss
            # with K.name_scope('action_loss'):
            #     loss_class_real = K.mean(K.sparse_categorical_crossentropy(
            #         _get_tensor(self.place_holders, 'true_label'),
            #         _get_tensor(self.real_outputs, 'label_out'), True))
            #     loss_class_fake = K.mean(K.sparse_categorical_crossentropy(
            #         _get_tensor(self.place_holders, 'true_label'),
            #         _get_tensor(self.fake_outputs, 'label_out'), True))
            #     disc_losses['disc_loss_action'] = self.action_scale_d * (loss_class_real + loss_class_fake)
            #     gen_losses['gen_loss_action'] = self.action_scale_g * loss_class_fake

            if self.coherence_loss:
                with K.name_scope('coherence_loss'):
                    exp_decay = 1.0 / np.exp(np.linspace(0.0, 5.0, self.seq_len // 2, dtype='float32'))
                    exp_decay = np.reshape(exp_decay, (1, 1, self.seq_len // 2, 1))
                    coherence_mask = np.concatenate([np.zeros((1, 1, self.seq_len // 2, 1)), exp_decay], axis=2)
                    coherence_mask = K.constant(coherence_mask, dtype='float32')
                    loss_coh = K.sum(K.mean(K.square((real_seq - gen_seq) * coherence_mask), axis=-1), axis=(1, 2))
                    gen_losses['gen_loss_coh'] = self.coherence_scale * K.mean(loss_coh)

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

            if self.smoothing_loss:
                with K.name_scope('smoothing_loss'):
                    Q = idct(np.eye(self.seq_len))[:self.smoothing_basis, :]
                    Q_inv = pinv(Q)
                    Qs = K.constant(np.matmul(Q_inv, Q), dtype='float32')
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

        return wgan_losses, disc_losses, gen_losses, gen_metrics

    def _pseudo_build_model(self, model, optimizer):
        # This function mimics compilation to enable saving the model
        model.optimizer = optimizer
        model.sample_weight_mode = None
        model.loss = 'custom_loss'
        model.loss_weights = None
        model.metrics = None
        return model

    def _proc_disc_inputs(self, input_tensors):
        scope = Scoping.get_global_scope()
        with scope.name_scope('discriminator'):

            x = _get_tensor(input_tensors, 'real_seq')

            if self.translate_start:
                x, self.stats[scope+'start_coords'] = translate_start_in(x)
            if self.rotate_start:
                x, self.stats[scope+'start_rotation'] = rotate_start_in(x, self.body_members)
            if self.rescale_coords:
                x, self.stats[scope+'bone_len'] = rescale_body_in(x, self.body_members)

            self.org_shape = [int(dim) for dim in x.shape]

        return x

    def _proc_disc_outputs(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('discriminator'):
            with scope.name_scope('score_net'):
                n_hidden = 256
                x = Dense(n_hidden, name=scope+'dense_in', activation='relu')(x)
                for i in range(3):
                    with scope.name_scope('block_%d' % i):
                        pi = Dense(n_hidden, name=scope+'dense_0', activation='relu')(x)
                        pi = Dense(n_hidden, name=scope+'dense_1', activation='relu')(pi)

                        x = Add(name=scope+'add')([x, pi])

                score_out = Dense(1, name=scope+'score_out')(x)

            output_tensors = [score_out]

            # label_out = Dense(self.num_actions, name=scope+'label_out')(x)
            # output_tensors.append(label_out)

        return output_tensors

    def _proc_gen_inputs(self, input_tensors):
        scope = Scoping.get_global_scope()
        with scope.name_scope('generator'):

            x = _get_tensor(input_tensors, 'real_seq')
            x_mask = _get_tensor(input_tensors, 'seq_mask')

            if self.translate_start:
                x, self.stats[scope+'start_coords'] = translate_start_in(x)
            if self.rotate_start:
                x, self.stats[scope+'start_rotation'] = rotate_start_in(x, self.body_members)
                # self.aux_out = x  # Uncomment to visualize rotated sequence
                # self.aux_out = rotate_start_out(x)  # Uncomment to visualize re-rotated sequence
            if self.rescale_coords:
                x, self.stats[scope+'bone_len'] = rescale_body_in(x, self.body_members)
            if self.remove_hip:
                x, x_mask, self.stats[scope+'hip_info'] = remove_hip_in(x, x_mask, self.data_set)
            if self.use_diff:
                x, x_mask, self.stats[scope+'start_pose'] = seq_to_diff_in(x, x_mask)
                self.diff_input, self.diff_mask = x, x_mask
            if self.use_angles:
                x, x_mask, self.stats[scope+'hip_coords'],  self.stats[scope+'bone_len'], self.stats[scope+'fixed_angles'] = \
                    seq_to_angles_in(x, x_mask, self.body_members)
                self.angles_input, self.angles_mask = x, x_mask
                # self.aux_out = seq_to_angles_out(x)  # Uncomment to visualize reconstructed sequence

            self.org_shape = [int(dim) for dim in x.shape]

            x = Multiply(name=scope+'mask_mult')([x, x_mask])
            # x_occ = Lambda(lambda arg: 1 - arg, name=scope+'mask_occ')(x_mask)
            x = Concatenate(axis=-1, name=scope+'cat_occ')([x, x_mask])

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
            num_actions = self.num_actions
            if self.action_cond:
                x_label = _get_tensor(input_tensors, 'true_label')
                x_label = Lambda(lambda arg: K.one_hot(arg, num_actions), name=scope+'emb_label')(x_label)
                x_label = Reshape((1, 1, num_actions), name=scope+'res_label')(x_label)
                x_label = Tile((x.shape[1], x.shape[2], 1), name=scope+'tile_label')(x_label)
                x = Concatenate(axis=-1, name=scope+'cat_label')([x, x_label])

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
                x = seq_to_angles_out(x, self.body_members, self.stats[scope+'hip_coords'],
                                      self.stats[scope+'bone_len'],  self.stats[scope+'fixed_angles'])
            if self.use_diff:
                self.diff_output = x
                x = seq_to_diff_out(x, self.stats[scope+'start_pose'])
            if self.remove_hip:
                x = remove_hip_out(x, self.stats[scope+'hip_info'], self.data_set)
            if self.rescale_coords:
                x = rescale_body_out(x, self.stats[scope+'bone_len'])
            if self.rotate_start:
                x = rotate_start_out(x, self.stats[scope+'start_rotation'])
            if self.translate_start:
                x = translate_start_out(x, self.stats[scope+'start_coords'])

            output_tensors = [x]

        return output_tensors

    def _pose_encoder(self, seq):
        scope = Scoping.get_global_scope()
        with scope.name_scope('encoder'):
            fae_dim = self.org_shape[1] * self.org_shape[3] * 4

            h = Permute((2, 1, 3), name=scope+'perm_in')(seq)
            h = Reshape((int(seq.shape[2]), int(seq.shape[1] * seq.shape[3])), name=scope+'resh_in')(h)

            h = Conv1D(fae_dim, 1, 1, name=scope+'conv_in', **CONV1D_ARGS)(h)
            for i in range(3):
                with scope.name_scope('block_%d' % i):
                    pi = Conv1D(fae_dim, 1, 1, activation='relu', name=scope+'pi_0', **CONV1D_ARGS)(h)
                    pi = Conv1D(fae_dim, 1, 1, activation='relu', name=scope+'pi_1', **CONV1D_ARGS)(pi)
                    tau = Conv1D(fae_dim, 1, 1, activation='sigmoid', name=scope+'tau_0', **CONV1D_ARGS)(h)
                    h = Lambda(lambda args: (args[0] * (1 - args[2])) + (args[1] * args[2]),
                               name=scope+'attention')([h, pi, tau])

            z = Conv1D(fae_dim, 1, 1, name=scope+'z_emb', **CONV1D_ARGS)(h)
            z_attention = Conv1D(fae_dim, 1, 1, activation='sigmoid', name=scope+'attention_mask', **CONV1D_ARGS)(h)

            # We are only expecting half of the latent features to be activated
            z = Multiply(name=scope+'z_attention')([z, z_attention])

        return z

    def _pose_decoder(self, gen_z):
        scope = Scoping.get_global_scope()
        with scope.name_scope('decoder'):
            fae_dim = self.org_shape[1] * self.org_shape[3] * 4

            dec_h = Conv1D(fae_dim, 1, 1, name=scope+'conv_in', **CONV1D_ARGS)(gen_z)
            for i in range(3):
                with scope.name_scope('block_%d' % i):
                    pi = Conv1D(fae_dim, 1, 1, activation='relu', name=scope+'pi_0', **CONV1D_ARGS)(dec_h)
                    pi = Conv1D(fae_dim, 1, 1, activation='relu', name=scope+'pi_1', **CONV1D_ARGS)(pi)
                    tau = Conv1D(fae_dim, 1, 1, activation='sigmoid', name=scope+'tau_0', **CONV1D_ARGS)(dec_h)
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
    x = conv_func(filters=out_filters // bneck_factor, kernel_size=kernel_size, strides=1,
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
                fae_dim = self.org_shape[1] * self.org_shape[3] * 4
                x = Dense((self.seq_len * fae_dim), name=scope+'dense_out', activation='relu')(x)
                x = Reshape((self.seq_len, fae_dim, 1), name=scope+'reshape_out')(x)
            else:
                x = Dense((self.njoints * self.seq_len * 3), name=scope+'dense_out', activation='relu')(x)
                x = Reshape((self.njoints, self.seq_len, 3), name=scope+'reshape_out')(x)

        return x


def resnet_disc(x):
    scope = Scoping.get_global_scope()
    with scope.name_scope('resnet'):
        n_hidden = 16
        block_factors = [1, 2, 4]
        block_strides = [2, 2, 2]
        n_reps = 2

        x = Conv2D(n_hidden * block_factors[0], 3, 1, name=scope+'conv_in', **CONV2D_ARGS)(x)
        for i, factor in enumerate(block_factors):
            for j in range(n_reps):
                with scope.name_scope('block_%d_%d' % (i, j)):
                    n_filters = n_hidden * factor
                    strides = block_strides[i] if j == 0 else 1
                    if int(x.shape[-1]) != n_filters or strides > 1:
                        shortcut = Conv2D(n_filters, strides, block_strides[i],
                                          name=scope + 'shortcut', **CONV2D_ARGS)(x)
                    else:
                        shortcut = x
                    pi = _conv_block(x, n_filters, 2, 3, strides)
                    x = Add(name=scope+'add')([shortcut, pi])

        x = Activation('relu', name=scope+'relu_out')(x)
        x = Lambda(lambda x: K.mean(x, axis=(1, 2)), name=scope+'mean_pool')(x)
    return x


def dmnn_disc(x):
    scope = Scoping.get_global_scope()
    with scope.name_scope('dmnn'):
        x_shape = [int(dim) for dim in x.shape]
        blocks = [{'size': 16, 'bneck_f': 2, 'strides': 2},
                  {'size': 32, 'bneck_f': 2, 'strides': 2},
                  {'size': 64, 'bneck_f': 2, 'strides': 2}]
        n_reps = 2

        x = CombMatrix(x_shape[1], name=scope+'comb_matrix')(x)

        x = EDM(name=scope+'edms')(x)
        x = Reshape((x_shape[1], x_shape[1], x_shape[2], 1), name=scope+'resh_in')(x)

        x = Conv3D(blocks[0]['size'] // blocks[0]['size'], 1, 1, name=scope+'conv_in', **CONV2D_ARGS)(x)
        for i in range(len(blocks)):
            for j in range(n_reps):
                with scope.name_scope('block_%d_%d' % (i, j)):
                    strides = blocks[i]['strides'] if j == 0 else 1
                    if int(x.shape[-1]) != blocks[i]['size'] or strides > 1:
                        shortcut = Conv3D(blocks[i]['size'], 1, strides,
                                          name=scope+'shortcut', **CONV2D_ARGS)(x)
                    else:
                        shortcut = x

                    x = _conv_block(x, blocks[i]['size'], blocks[i]['bneck_f'], 3, strides, Conv3D, 1)
                    x = Add(name=scope+'add')([shortcut, x])

        x = Lambda(lambda args: K.mean(args, axis=(1, 2, 3)), name=scope+'mean_pool')(x)
    return x


def split_disc(x):
    scope = Scoping.get_global_scope()
    x_shape = [int(dim) for dim in x.shape]
    time_steps = x_shape[2] // 2

    split_input = Input(batch_shape=(x_shape[0], x_shape[1], time_steps, x_shape[3]))
    split_x = Concatenate(axis=-1, name=scope+'split_features_cat')([resnet_disc(split_input), dmnn_disc(split_input)])
    split_res = Model(split_input, split_x, name='split_disc_model')

    x_head = Lambda(lambda arg: arg[:, :, :time_steps, :], name=scope+'head_slice')(x)
    x_tail = Lambda(lambda arg: arg[:, :, time_steps:, :], name=scope+'tail_slice')(x)

    x = Concatenate(axis=-1, name=scope+'features_cat')([split_res(x_head), split_res(x_tail)])
    return x


class MotionGANV5(_MotionGAN):
    # DMNN + ResNet + Split Discriminator, WaveNet style generator

    def discriminator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('discriminator'):
            x = split_disc(x)
        return x

    def generator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('generator'):
            x_shape = [int(dim) for dim in x.shape]
            n_hidden = 32
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
                        pi = _conv_block(wave_output, n_filters, 2, (3, 9), (2, 1), Conv2D)
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


class MotionGANV7(_MotionGAN):
    # DMNN + ResNet + Split Discriminator, ResNet + UNet Generator 4 STACK

    def discriminator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('discriminator'):
            x = split_disc(x)
        return x

    def generator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('generator'):
            n_hidden = 16
            u_blocks = 0
            emb_dim = int(x.shape[2])
            while emb_dim > 4:
                emb_dim //= 2
                u_blocks += 1
            u_blocks = min(u_blocks, 4)
            u_blocks = u_blocks * 2
            block_factors = range(1, (u_blocks // 2) + 1) + range(u_blocks // 2, 0, -1)
            macro_blocks = 4

            if not (self.time_pres_emb or self.use_pose_fae):
                x = Dense(4 * 4 * n_hidden * block_factors[0], name=scope+'dense_in')(x)
                x = Reshape((4, 4, n_hidden * block_factors[0]), name=scope+'reshape_in')(x)

            u_skips = []
            x = Conv2D(n_hidden, 1, 1, name=scope+'conv_in', **CONV2D_ARGS)(x)
            for k in range(macro_blocks):
                with scope.name_scope('macro_block_%d' % k):
                    shortcut = x
                    for i, factor in enumerate(block_factors):
                        with scope.name_scope('block_%d' % i):
                            n_filters = n_hidden * factor
                            if i < (u_blocks // 2):
                                conv_func = Conv2D
                                u_skips.append(x)
                            else:
                                conv_func = Conv2DTranspose

                            with scope.name_scope('pi'):
                                x = _conv_block(x, n_filters, 2, 3, (1, 2), conv_func)

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

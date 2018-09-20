from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras.api.keras.backend as K
from scipy.fftpack import idct
from scipy.linalg import pinv
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Input
from tensorflow.contrib.keras.api.keras.layers import Conv2DTranspose, Conv2D, \
    Dense, Activation, Lambda, Add, Concatenate, Permute, Reshape, Flatten, \
    Conv1D, Multiply, Cropping2D, Embedding
from tensorflow.contrib.keras.api.keras.optimizers import Nadam
from tensorflow.contrib.keras.api.keras.regularizers import l2
from layers.edm import edm, EDM
from layers.comb_matrix import CombMatrix
from layers.tile import Tile
from layers.seq_transform import remove_hip_in, remove_hip_out, translate_start_in, translate_start_out, \
    rotate_start_in, rotate_start_out, rescale_body_in, rescale_body_out, \
    seq_to_diff_in, seq_to_diff_out, seq_to_angles_in, seq_to_angles_out
from layers.relational_memory import RelationalMemoryRNN
from layers.causal_conv import CausalConv1D, CausalConv2D
from layers.normalization import InstanceNormalization
from layers.cudnn_recurrent import CuDNNLSTM
from collections import OrderedDict
from utils.scoping import Scoping

CONV1D_ARGS = {'padding': 'same', 'kernel_regularizer': l2(5e-4)}
CONV2D_ARGS = {'padding': 'same', 'data_format': 'channels_last', 'kernel_regularizer': l2(5e-4)}


def get_model(config):
    class_name = 'MotionGANV' + config.model_version[1:]
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
        self.gan_type = config.gan_type
        self.gan_scale_d = 10.0 * config.loss_factor
        self.gan_scale_g = 10.0 * config.loss_factor * (0.0 if self.gan_type == 'no_gan' else 1.0)
        self.rec_scale = 1.0   # if 'expmaps' not in self.data_set else 10.0
        self.latent_cond_dim = config.latent_cond_dim
        self.latent_scale_d = 10.0
        self.latent_scale_g = 1.0
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
        self.translate_start = config.translate_start
        self.rotate_start = config.rotate_start
        self.rescale_coords = config.rescale_coords
        self.remove_hip = config.remove_hip
        self.use_diff = config.use_diff
        self.diff_scale = 100.0
        self.use_angles = config.use_angles
        self.angles_scale = 0.5
        self.last_known = config.last_known
        self.add_skip = config.add_skip
        self.no_dmnn_disc = config.no_dmnn_disc

        self.stats = {}
        self.z_params = []

        # Discriminator
        real_seq = Input(batch_shape=(self.batch_size, self.njoints, self.seq_len, 3), name='real_seq', dtype='float32')
        self.disc_inputs = [real_seq]
        self.place_holders = []
        if self.action_cond:
            true_label = Input(batch_shape=(self.batch_size, 1), name='true_label', dtype='int32')
            self.place_holders.append(true_label)  # it is not an input because it is only used in the loss
        if self.latent_cond_dim > 0:
            latent_cond = Input(batch_shape=(self.batch_size, self.latent_cond_dim), name='latent_cond', dtype='float32')
            self.place_holders.append(latent_cond)
        x = self._proc_disc_inputs(self.disc_inputs)
        self.real_outputs = self._proc_disc_outputs(self.discriminator(x))
        self.disc_model = Model(self.disc_inputs, self.real_outputs, name=self.name + '_discriminator')

        # Generator
        seq_mask = Input(batch_shape=(self.batch_size, self.njoints, self.seq_len, 1), name='seq_mask', dtype='float32')
        self.gen_inputs = [real_seq, seq_mask]
        if self.action_cond:
            self.gen_inputs.append(true_label)
        if self.latent_cond_dim > 0:
            self.gen_inputs.append(latent_cond)
        x = self._proc_gen_inputs(self.gen_inputs)
        self.gen_outputs = self._proc_gen_outputs(self.generator(x))
        self.gen_model = Model(self.gen_inputs, self.gen_outputs, name=self.name + '_generator')
        self.fake_outputs = self.disc_model(self.gen_outputs)

        # Losses
        self.gan_losses, self.disc_losses, self.gen_losses, self.gen_metrics = self._build_loss()

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
                                           self.gan_losses.values() + self.disc_losses.values(),
                                           disc_training_updates)

        with K.name_scope('discriminator/functions/eval'):
            self.disc_eval_f = K.function(self.disc_inputs + self.gen_inputs,
                                          self.gan_losses.values() + self.disc_losses.values())

        self.disc_model = self._pseudo_build_model(self.disc_model, disc_optimizer)

        with K.name_scope('generator/functions/train'):
            gen_optimizer = Nadam(lr=config.learning_rate)
            gen_training_updates = gen_optimizer.get_updates(gen_loss, self.gen_model.trainable_weights)
            self.gen_train_f = K.function(self.gen_inputs,
                                          self.gen_losses.values() + self.gen_metrics.values(),
                                          gen_training_updates)

        with K.name_scope('generator/functions/eval'):
            gen_f_outs = self.gen_losses.values() + self.gen_metrics.values()
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
        keys = self.gan_losses.keys() + self.disc_losses.keys()
        keys = ['train/%s' % key for key in keys]
        losses_dict = OrderedDict(zip(keys, train_outs))
        return losses_dict

    def disc_eval(self, inputs):
        eval_outs = self.disc_eval_f(inputs)
        keys = self.gan_losses.keys() + self.disc_losses.keys()
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
            gan_losses = OrderedDict()
            disc_losses = OrderedDict()
            gen_losses = OrderedDict()
            gen_metrics = OrderedDict()

            # Grabbing tensors
            real_seq = _get_tensor(self.disc_inputs, 'real_seq')
            seq_mask = _get_tensor(self.gen_inputs, 'seq_mask')
            seq_mask = seq_mask if not self.gan_type == 'no_gan' else K.ones_like(seq_mask)
            gen_seq = self.gen_outputs[0]

            no_zero_frames = K.cast(K.greater_equal(K.abs(K.sum(real_seq, axis=(1, 3))), K.epsilon()), 'float32')
            no_zero_frames_edm = K.reshape(no_zero_frames, (no_zero_frames.shape[0], 1, 1, no_zero_frames.shape[1]))

            if self.gan_type == 'wgan':
                # WGAN Basic losses
                with K.name_scope('gan_loss'):
                    loss_real = _get_tensor(self.real_outputs, 'score_out')
                    loss_fake = _get_tensor(self.fake_outputs, 'score_out')
                    gan_losses['loss_real'] = K.mean(loss_real)  # K.mean(K.abs(loss_real))
                    gan_losses['loss_fake'] = K.mean(loss_fake)  # K.mean(K.abs(loss_fake))

                    # Interpolates for GP
                    alpha = K.random_uniform((self.batch_size, 1, 1, 1))
                    interpolates = (alpha * real_seq) + ((1 - alpha) * gen_seq)

                    # Gradient Penalty
                    inter_outputs = self.disc_model(interpolates)
                    inter_score = _get_tensor(inter_outputs, 'score_out')
                    grad_mixed = K.gradients(inter_score, [interpolates])[0]
                    norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=(1, 2, 3)) + K.epsilon())
                    grad_penalty = K.expand_dims(K.square(norm_grad_mixed - self.gamma_grads) / (self.gamma_grads ** 2), axis=-1)

                    # WGAN-GP losses
                    disc_loss_gan = loss_fake - loss_real + (self.lambda_grads * grad_penalty)  # -K.abs(loss_real - loss_fake) + (K.square(loss_real) * 0.1) + (self.lambda_grads * grad_penalty)
                    disc_losses['disc_loss_gan'] = self.gan_scale_d * K.mean(disc_loss_gan)

                    gen_loss_gan = -loss_fake  # K.abs(loss_real - loss_fake) + (K.square(loss_fake) * 0.1)
                    gen_losses['gen_loss_gan'] = self.gan_scale_g * K.mean(gen_loss_gan)

            elif self.gan_type == 'standard':
                # GAN Basic losses
                with K.name_scope('gan_loss'):
                    Kone = K.ones((self.batch_size, 1), dtype='float32')
                    Kzero = K.zeros((self.batch_size, 1), dtype='float32')
                    loss_real = K.binary_crossentropy(Kone, _get_tensor(self.real_outputs, 'score_out'), True)
                    loss_fake = K.binary_crossentropy(Kzero, _get_tensor(self.fake_outputs, 'score_out'), True)
                    gan_losses['loss_real'] = K.mean(loss_real)
                    gan_losses['loss_fake'] = K.mean(loss_fake)

                    # R1 Gradient Penalty
                    grad_disc = K.gradients(_get_tensor(self.real_outputs, 'score_out'), self.disc_inputs)[0]
                    norm_grad_disc = K.sum(K.square(grad_disc), axis=(1, 2, 3))

                    # GAN-GP losses
                    disc_loss_gan = loss_real + loss_fake + (self.lambda_grads * norm_grad_disc)
                    disc_losses['disc_loss_gan'] = self.gan_scale_d * K.mean(disc_loss_gan)

                    gen_loss_gan = K.binary_crossentropy(Kone, _get_tensor(self.fake_outputs, 'score_out'), True)
                    gen_losses['gen_loss_gan'] = self.gan_scale_g * K.mean(gen_loss_gan)

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

                loss_rec = K.sum(K.mean(K.square((real_seq - gen_seq) * (1 - _get_tensor(self.gen_inputs, 'seq_mask'))), axis=-1), axis=(1, 2))
                gen_metrics['gen_loss_rec_comp'] = self.rec_scale * K.mean(loss_rec)

            # with K.name_scope('kl_loss'):
            #     kl_loss_sum = 0
            #     for z_mean, z_log_var in self.z_params:
            #         kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            #         kl_loss = K.sum(kl_loss, axis=-1)
            #         kl_loss *= -0.5
            #         kl_loss_sum += kl_loss
            #     gen_metrics['gen_loss_kl'] = K.mean(kl_loss_sum)

            # Action label loss
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
                    loss_latent_real = K.mean(K.square(
                        _get_tensor(self.place_holders, 'latent_cond') - _get_tensor(self.real_outputs, 'latent_out')))
                    loss_latent_fake = K.mean(K.square(
                        _get_tensor(self.place_holders, 'latent_cond') - _get_tensor(self.fake_outputs, 'latent_out')))
                    disc_losses['disc_loss_latent'] = self.latent_scale_d * (loss_latent_real + loss_latent_fake)
                    gen_losses['gen_loss_latent'] = self.latent_scale_g * loss_latent_fake

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
                    gen_losses['gen_loss_limbs'] = self.shape_scale * 0.01 * K.mean(loss_shape)

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

        return gan_losses, disc_losses, gen_losses, gen_metrics

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
            def _out_net(x_out, n_out, net_name, activation=None):
                with scope.name_scope(net_name+'_net'):
                    return Dense(n_out, name=scope+net_name+'_out', activation=activation)(x_out)

            output_tensors = [_out_net(x, 1, 'score')]

            if self.action_cond:
                output_tensors.append(_out_net(x, self.num_actions, 'label'))

            if self.latent_cond_dim > 0:
                with scope.name_scope('latent_net'):
                    output_tensors.append(_out_net(x, self.latent_cond_dim, 'latent'))

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

            if self.last_known:
                known_size = self.org_shape[2] // 2
                x = Lambda(lambda arg: arg[:, :, :known_size, :], name=scope+'slice_unk')(x)
                x_last = Lambda(lambda arg: arg[:, :, -1, :], name=scope+'last_known')(x)
                x_last = Reshape((self.org_shape[1], 1, self.org_shape[3]), name=scope+'res_last_known')(x_last)
                x_last = Tile((1, known_size, 1), name=scope+'tl_last_known')(x_last)
                x = Concatenate(axis=2, name=scope+'cat_last_known')([x, x_last])

            if self.add_skip:
                self.org_coords = x

            x_mask = Lambda(lambda arg: 1 - arg, name=scope+'mask_occ')(x_mask)
            x = Concatenate(axis=-1, name=scope+'cat_occ')([x, x_mask])

            x = self._pose_encoder(x)
            self.fae_z = Reshape((int(x.shape[1]), int(x.shape[2])), name=scope+'fae_z_reshape')(x)

            if self.action_cond:
                num_actions = self.num_actions
                emb_dim = 3
                x_label = _get_tensor(self.gen_inputs, 'true_label')
                x_label = Embedding(num_actions, emb_dim, name=scope+'emb_label')(x_label)
                x_label = Reshape((1, 1, emb_dim), name=scope+'res_label')(x_label)
                x_label = Tile((x.shape[1], x.shape[2], 1), name=scope+'tile_label')(x_label)
                x = Concatenate(axis=-1, name=scope+'cat_label')([x, x_label])

            if self.latent_cond_dim > 0:
                x_latent = _get_tensor(self.gen_inputs, 'latent_cond')
                x_latent = Reshape((1, 1, self.latent_cond_dim), name=scope+'res_latent')(x_latent)
                x_latent = Tile((x.shape[1], x.shape[2], 1), name=scope+'tile_latent')(x_latent)
                x = Concatenate(axis=-1, name=scope + 'cat_latent')([x, x_latent])

        return x

    def _proc_gen_outputs(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('generator'):

            x = self._pose_decoder(x)
            # self.fae_gen_z = Reshape((int(x.shape[1]), int(x.shape[2])), name=scope+'fae_gen_z_reshape')(x)

            if self.add_skip:
                x = Add(name=scope + 'add_coords_out')([self.org_coords, x])
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
                    pi = Activation('relu', name=scope+'relu_0')(h)
                    pi = Conv1D(fae_dim // 2, 1, 1, name=scope+'pi_0', **CONV1D_ARGS)(pi)
                    pi = Activation('relu', name=scope+'relu_1')(pi)
                    pi = Conv1D(fae_dim, 1, 1, name=scope+'pi_1', **CONV1D_ARGS)(pi)
                    tau = Conv1D(fae_dim, 1, 1, activation='sigmoid', name=scope+'tau_0', **CONV1D_ARGS)(h)
                    h = Lambda(lambda args: (args[0] * (1 - args[2])) + (args[1] * args[2]),
                               name=scope+'attention')([h, pi, tau])

            z = Conv1D(fae_dim, 1, 1, name=scope+'z', **CONV1D_ARGS)(h)
            z_attention = Conv1D(fae_dim, 1, 1, activation='sigmoid',
                                 name=scope+'z_attention', **CONV1D_ARGS)(h)

            # We are only expecting half of the latent features to be activated
            z = Multiply(name=scope+'z_attended')([z, z_attention])

            # self.z_params.append((z_mean, z_log_var))
            # reparameterization trick
            # instead of sampling from Q(z|X), sample eps = N(0,I)
            # z = z_mean + sqrt(var)*eps
            # def sampling(args):
            #     """Reparameterization trick by sampling fr an isotropic unit Gaussian.
            #     # Arguments:
            #         args (tensor): mean and log of variance of Q(z|X)
            #     # Returns:
            #         z (tensor): sampled latent vector
            #     """
            #
            #     z_mean, z_log_var = args
            #     # by default, random_normal has mean=0 and std=1.0
            #     epsilon = K.random_normal(shape=[int(dim) for dim in z_mean.shape])
            #
            #     return z_mean + K.exp(0.5 * z_log_var) * epsilon
            #
            # z = Lambda(sampling, output_shape=(fae_dim,), name=scope+'z')([z_mean, z_log_var])
            z = Reshape((int(z.shape[1]), int(z.shape[2]), 1), name=scope+'res_out')(z)

        return z

    def _pose_decoder(self, gen_z):
        scope = Scoping.get_global_scope()
        with scope.name_scope('decoder'):
            dec_h = Conv2D(1, 3, 1, name=scope+'fae_merge', **CONV2D_ARGS)(gen_z)
            dec_h = Reshape((int(gen_z.shape[1]), int(gen_z.shape[2])), name=scope+'fae_reshape')(dec_h)

            fae_dim = self.org_shape[1] * self.org_shape[3] * 4

            dec_h = Conv1D(fae_dim, 1, 1, name=scope+'conv_in', **CONV1D_ARGS)(dec_h)
            for i in range(3):
                with scope.name_scope('block_%d' % i):
                    pi = Activation('relu', name=scope+'relu_0')(dec_h)
                    pi = Conv1D(fae_dim // 2, 1, 1, name=scope+'pi_0', **CONV1D_ARGS)(pi)
                    pi = Activation('relu', name=scope+'relu_1')(pi)
                    pi = Conv1D(fae_dim, 1, 1, name=scope+'pi_1', **CONV1D_ARGS)(pi)
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

            for i, factor in enumerate(block_factors):
                with scope.name_scope('block_%d' % i):
                    n_filters = n_hidden * factor
                    pi = _conv_block(x, n_filters, 1, 3, 1, Conv2DTranspose)

                    if i < self.nblocks - 1:
                        shortcut = Conv2DTranspose(n_filters, 1, 1,
                                                   name=scope+'shortcut', **CONV2D_ARGS)(x)
                        x = Add(name=scope+'add')([shortcut, pi])
                    else:
                        x = pi

            # x = InstanceNormalization(axis=-1, name=scope+'inorm_out')(x)
            # x = Activation('relu', name=scope+'relu_out')(x)

        return x


def _preact_dense(x, n_units):
    scope = Scoping.get_global_scope()
    # x = InstanceNormalization(name=scope+'inorm')(x)
    x = Activation('relu', name=scope+'relu')(x)
    x = Dense(n_units, name=scope+'dense', activation='relu')(x)
    return x


class MotionGANV3(_MotionGAN):
    # Simple FeedForward with Residual Connections

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

            x_shape = [int(dim) for dim in x.shape]
            x = Flatten(name=scope+'flatten_in')(x)
            x = _preact_dense(x, n_hidden)
            for i in range(4):
                with scope.name_scope('block_%d' % i):
                    with scope.name_scope('pi_0'):
                        pi = _preact_dense(x, n_hidden)
                    with scope.name_scope('pi_1'):
                        pi = _preact_dense(pi, n_hidden)

                    x = Add(name=scope+'add')([x, pi])

            x = Dense((x_shape[1] * x_shape[2]), name=scope+'dense_out', activation='relu')(x)
            x = Reshape((x_shape[1], x_shape[2], 1), name=scope+'reshape_out')(x)

        return x


def resnet_disc(x):
    scope = Scoping.get_global_scope()
    with scope.name_scope('resnet'):
        n_hidden = 64
        n_reps = 2
        n_blocks = 3

        x = Conv2D(n_hidden, 1, 1, name=scope+'conv_in', **CONV2D_ARGS)(x)
        x_outs = []
        for i in range(n_blocks):
            for j in range(n_reps):
                with scope.name_scope('block_%d_%d' % (i, j)):
                    n_filters = n_hidden * (2 ** i)
                    strides = 2 if j == 0 else 1
                    if int(x.shape[-1]) != n_filters or strides > 1:
                        shortcut = Conv2D(n_filters, 1, strides,
                                          name=scope+'shortcut', **CONV2D_ARGS)(x)
                    else:
                        shortcut = x
                    with scope.name_scope('pi'):
                        pi = _conv_block(x, n_filters, 4, 3, strides)
                    x = Add(name=scope+'add')([shortcut, pi])
                    if j == n_reps - 1:
                        x_out = Reshape((int(x.shape[1]) * int(x.shape[2]),
                                         int(x.shape[3])), name=scope+'res_out')(x)
                        x_out = Activation('relu', name=scope+'relu_out')(x_out)
                        x_out = Conv1D(n_hidden * 2, 1, 1, name=scope+'dense_out', **CONV1D_ARGS)(x_out)
                        x_out = Lambda(lambda arg: K.mean(arg, axis=1), name=scope+'mean_pool')(x_out)
                        x_outs.append(x_out)

        # x = Activation('relu', name=scope+'relu_out')(x)
        # # x = Flatten(name=scope+'flatten_out')(x)
        # x = Lambda(lambda x: K.mean(x, axis=(1, 2)), name=scope+'mean_pool')(x)
        x = Concatenate(axis=-1, name=scope+'cat_out')(x_outs)
    return x


def dmnn_disc(x):
    scope = Scoping.get_global_scope()
    with scope.name_scope('dmnn'):
        x_shape = [int(dim) for dim in x.shape]
        n_hidden = 64
        n_reps = 2
        n_blocks = 3

        x = CombMatrix(x_shape[1], name=scope+'comb_matrix')(x)

        x = EDM(name=scope+'edms')(x)
        x = Reshape((x_shape[1] * x_shape[1], x_shape[2], 1), name=scope+'resh_in')(x)

        x = Conv2D(n_hidden, 1, 1, name=scope+'conv_in', **CONV2D_ARGS)(x)
        x_outs = []
        for i in range(n_blocks):
            for j in range(n_reps):
                with scope.name_scope('block_%d_%d' % (i, j)):
                    n_filters = n_hidden * (2 ** i)
                    strides = 2 if j == 0 else 1
                    if int(x.shape[-1]) != n_filters or strides > 1:
                        shortcut = Conv2D(n_filters, 1, strides,
                                          name=scope+'shortcut', **CONV2D_ARGS)(x)
                    else:
                        shortcut = x
                    with scope.name_scope('pi'):
                        pi = _conv_block(x, n_filters, 4, 3, strides)
                    x = Add(name=scope+'add')([shortcut, pi])
                    if j == n_reps - 1:
                        x_out = Reshape((int(x.shape[1]) * int(x.shape[2]),
                                         int(x.shape[3])), name=scope + 'res_out')(x)
                        x_out = Activation('relu', name=scope+'relu_out')(x_out)
                        x_out = Conv1D(n_hidden * 2, 1, 1, name=scope+'dense_out', **CONV1D_ARGS)(x_out)
                        x_out = Lambda(lambda arg: K.mean(arg, axis=1), name=scope+'mean_pool')(x_out)
                        x_outs.append(x_out)

        # x = Activation('relu', name=scope+'relu_out')(x)
        # # x = Flatten(name=scope+'flatten_out')(x)
        # x = Lambda(lambda x: K.mean(x, axis=(1, 2)), name=scope+'mean_pool')(x)
        x = Concatenate(axis=-1, name=scope+'cat_out')(x_outs)
    return x


def motion_disc(x):
    scope = Scoping.get_global_scope()
    with scope.name_scope('motion'):
        x_diff = Lambda(lambda arg: arg[:, :, 1:, :] - arg[:, :, :-1, :], name=scope+'x_diff')(x)
        x_diff = Permute((2, 1, 3), name=scope+'perm_diff')(x_diff)
        x_diff = Reshape((int(x_diff.shape[1]), 1, int(x_diff.shape[2] * x_diff.shape[3])), name=scope+'resh_diff')(x_diff)

        x_diff_edm = EDM(name=scope+'edms')(x)
        x_diff_edm = Lambda(lambda arg: arg[:, :, :, 1:] - arg[:, :, :, :-1], name=scope+'x_diff_edm')(x_diff_edm)
        x_diff_edm = Permute((3, 1, 2), name=scope+'perm_diff_edm')(x_diff_edm)
        x_diff_edm = Reshape((int(x_diff_edm.shape[1]), 1, int(x_diff_edm.shape[2] * x_diff_edm.shape[3])),
                             name=scope+'resh_diff_edm')(x_diff_edm)

        x = Concatenate(axis=-1, name=scope+'cat_diffs')([x_diff, x_diff_edm])

        n_hidden = 256
        n_reps = 2
        n_blocks = 3

        x = Conv2D(n_hidden, 1, 1, name=scope + 'conv_in', **CONV2D_ARGS)(x)
        for i in range(n_blocks):
            for j in range(n_reps):
                with scope.name_scope('block_%d_%d' % (i, j)):
                    strides = 2 if j == 0 else 1
                    if int(x.shape[-1]) != n_hidden or strides > 1:
                        shortcut = Conv2D(n_hidden, 1, strides,
                                          name=scope + 'shortcut',
                                          **CONV2D_ARGS)(x)
                    else:
                        shortcut = x
                    with scope.name_scope('pi'):
                        pi = _conv_block(x, n_hidden, 4, (3, 1), strides)
                    x = Add(name=scope + 'add')([shortcut, pi])

        x = Activation('relu', name=scope+'relu_out')(x)
        x = Lambda(lambda x: K.mean(x, axis=(1, 2)), name=scope+'mean_pool')(x)
    return x


def double_disc(x, no_dmnn_disc):
    scope = Scoping.get_global_scope()
    with scope.name_scope('discriminator'):
        features = [resnet_disc(x)] if no_dmnn_disc else [resnet_disc(x), dmnn_disc(x), motion_disc(x)]
        x = Concatenate(axis=-1, name=scope+'features_cat')(features)
    return x


class MotionGANV5(_MotionGAN):
    # DMNN + ResNet Discriminator, WaveNet style generator

    def discriminator(self, x):
        return double_disc(x, self.no_dmnn_disc)

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

            with scope.name_scope('wave_gen'):
                wave_input = Input(batch_shape=(x_shape[0], x_shape[1] // 2, x_shape[2], x_shape[3]))
                # print(time_steps, n_blocks)

                wave_output = wave_input
                for i in range(n_blocks):
                    with scope.name_scope('block_%d' % i):
                        n_filters = n_hidden * (i + 2)
                        shortcut = Conv2D(n_filters, (2, 1), (2, 1), name=scope+'shortcut', **CONV2D_ARGS)(wave_output)
                        with scope.name_scope('pi'):
                            pi = _conv_block(wave_output, n_filters, 2, 3, (2, 1), Conv2D)
                        wave_output = Add(name=scope+'add')([shortcut, pi])

                wave_output = Conv2D(1, 1, 1, name=scope+'merge_out', **CONV2D_ARGS)(wave_output)
                wave_output = Reshape((x_shape[2], 1), name=scope+'squeeze_out')(wave_output)

                wave_gen = Model(wave_input, wave_output, name='wave_gen_model')

            # print(wave_gen.summary())

            xs = []
            pred_x_labs = Lambda(lambda arg: K.stop_gradient(arg[:, x_shape[1] // 2, :, 1:]),
                                 name=scope+'pred_lab')(x)
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
                    pred_x = Concatenate(axis=-1, name=scope+'cat_pred_labs')([pred_x, pred_x_labs])
                    xs.append(pred_x)

            x = Lambda(lambda arg: arg[:, :x_shape[1] // 2, :, :], name=scope+'slice_out')(x)
            x = Reshape((x_shape[1] // 2, x_shape[2], x_shape[3]), name=scope+'res_slice_out')(x)
            xs = Lambda(lambda arg: K.stack(arg, axis=1), name=scope+'stack_out')(xs)
            xs = Reshape((x_shape[1] // 2, x_shape[2], x_shape[3]), name=scope+'res_stack_out')(xs)
            x = Concatenate(axis=1, name=scope+'cat_out')([x, xs])
            x = Lambda(lambda arg: arg[:, :, :, 0], name=scope+'trim_out')(x)
            x = Reshape((x_shape[1], x_shape[2], 1), name=scope+'res_trim_out')(x)

        return x


class MotionGANV7(_MotionGAN):
    # DMNN + ResNet Discriminator, ResNet + UNet Generator 4 STACK

    def discriminator(self, x):
        return double_disc(x, self.no_dmnn_disc)

    def generator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('generator'):
            n_hidden = 32
            u_blocks = 0
            min_dim = min(int(x.shape[1]), int(x.shape[2]))
            while min_dim > 4:
                min_dim //= 2
                u_blocks += 1
            u_blocks = min(u_blocks, 4)
            u_blocks = u_blocks * 2
            block_factors = range(1, (u_blocks // 2) + 1) + range(u_blocks // 2, 0, -1)
            macro_blocks = 4

            u_skips = []
            for k in range(macro_blocks):
                with scope.name_scope('macro_block_%d' % k):
                    x = Conv2D(n_hidden, 1, 1, name=scope+'conv_in', **CONV2D_ARGS)(x)
                    pi = x
                    for i, factor in enumerate(block_factors):
                        with scope.name_scope('block_%d' % i):
                            n_filters = n_hidden * factor
                            if i < (u_blocks // 2):
                                conv_func = Conv2D
                                u_skips.append(pi)
                            else:
                                conv_func = Conv2DTranspose

                            with scope.name_scope('pi'):
                                pi = _conv_block(pi, n_filters, 2, 3, 2, conv_func)

                            if (u_blocks // 2) <= i < u_blocks:
                                skip_pi = u_skips.pop()
                                if skip_pi.shape[1] != pi.shape[1] or skip_pi.shape[2] != pi.shape[2]:
                                    pi = Cropping2D(((0, int(pi.shape[1] - skip_pi.shape[1])),
                                                    (0, int(pi.shape[2] - skip_pi.shape[2]))),
                                                     name=scope+'crop_pi')(pi)
                                pi = Concatenate(name=scope+'cat_skip')([skip_pi, pi])
                                with scope.name_scope('skip_pi'):
                                    pi = _conv_block(pi, n_filters, 2, 3, 1, conv_func)

                    x = Add(name=scope+'add')([x, pi])
        return x


class MotionGANV8(_MotionGAN):
    # DMNN + ResNet Discriminator, LSTM + MHDPA Generator

    def discriminator(self, x):
        return double_disc(x, self.no_dmnn_disc)

    def generator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('generator'):

            chans = int(x.shape[3])
            x = Reshape((int(x.shape[1]), int(x.shape[2]) * chans), name=scope+'resh_in')(x)
            x = Conv1D(int(x.shape[2]) // (2 * chans), 1, 1, name=scope+'conv_in', **CONV1D_ARGS)(x)

            x_shape = [int(dim) for dim in x.shape]
            n_stages = 2
            for i in range(n_stages):
                with scope.name_scope('stage_%d' % i):
                    pi = RelationalMemoryRNN(8, x_shape[2], 8, return_sequences=True,
                                             name=scope+'pi_rel_mem')(x)
                    pi = Conv1D(x_shape[2], 1, 1, activation='relu',
                                name=scope + 'pi_conv_0', **CONV1D_ARGS)(pi)
                    pi = Conv1D(x_shape[2], 1, 1,
                                name=scope + 'pi_conv_1', **CONV1D_ARGS)(pi)
                    x = Add(name=scope + 'add')([x, pi])

            x = Reshape((x_shape[1], x_shape[2], 1), name=scope+'resh_out')(x)

        return x


class MotionGANV9(_MotionGAN):
    # DMNN + ResNet Discriminator, CausalConv Generator

    def discriminator(self, x):
        return double_disc(x, self.no_dmnn_disc)

    def generator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('generator'):
            n_hidden = 32
            u_blocks = 0
            min_dim = min(int(x.shape[1]), int(x.shape[2]))
            while min_dim > 4:
                min_dim //= 2
                u_blocks += 1
            u_blocks = min(u_blocks, 4)
            u_blocks = u_blocks * 2
            block_factors = range(1, (u_blocks // 2) + 1) + range(u_blocks // 2, 0, -1)
            macro_blocks = 4

            u_skips = []
            for k in range(macro_blocks):
                with scope.name_scope('macro_block_%d' % k):
                    x = Conv2D(n_hidden, 1, 1, name=scope+'conv_in', **CONV2D_ARGS)(x)
                    pi = x
                    for i, factor in enumerate(block_factors):
                        with scope.name_scope('block_%d' % i):
                            n_filters = n_hidden * factor
                            if i < (u_blocks // 2):
                                conv_func = CausalConv2D
                                u_skips.append(pi)
                            else:
                                conv_func = Conv2DTranspose

                            with scope.name_scope('pi'):
                                pi = _conv_block(pi, n_filters, 2, 3, 2, conv_func)

                            if (u_blocks // 2) <= i < u_blocks:
                                skip_pi = u_skips.pop()
                                if skip_pi.shape[1] != pi.shape[1] or skip_pi.shape[2] != pi.shape[2]:
                                    pi = Cropping2D(((0, int(pi.shape[1] - skip_pi.shape[1])),
                                                    (0, int(pi.shape[2] - skip_pi.shape[2]))),
                                                     name=scope+'crop_pi')(pi)
                                pi = Concatenate(name=scope+'cat_skip')([skip_pi, pi])
                                with scope.name_scope('skip_pi'):
                                    pi = _conv_block(pi, n_filters, 2, 3, 1, CausalConv2D)

                    x = Add(name=scope+'add')([x, pi])
        return x


class MotionGANV87(_MotionGAN):
    # Super GAN 87

    def discriminator(self, x):
        return double_disc(x, self.no_dmnn_disc)

    def generator(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('generator'):

            with scope.name_scope('RNN'):
                chans = int(x.shape[3])
                x = Reshape((int(x.shape[1]), int(x.shape[2]) * chans), name=scope+'resh_in')(x)
                x = Conv1D(int(x.shape[2]) // chans, 1, 1, name=scope+'conv_in', **CONV1D_ARGS)(x)

                x_shape = [int(dim) for dim in x.shape]

                n_stages = 1
                with scope.name_scope('rel_mem_rnn'):
                    for k in range(n_stages):
                        with scope.name_scope('stage_%d' % k):
                            pi = RelationalMemoryRNN(8, x_shape[2] // 4, 8, return_sequences=True,
                                                     name=scope+'pi_rel_mem')(x)
                            pi = Conv1D(x_shape[2], 1, 1, activation='relu',
                                        name=scope + 'pi_conv_0', **CONV1D_ARGS)(pi)
                            pi = Conv1D(x_shape[2], 1, 1,
                                        name=scope + 'pi_conv_1', **CONV1D_ARGS)(pi)
                            x = Add(name=scope + 'add')([x, pi])

                n_stages = 1
                with scope.name_scope('lstm'):
                    for k in range(n_stages):
                        with scope.name_scope('stage_%d' % k):
                            pi = CuDNNLSTM(x_shape[2] // 4, return_sequences=True, name=scope+'pi_lstm')(x)
                            pi = Conv1D(x_shape[2], 1, 1, activation='relu',
                                        name=scope+'pi_conv_0', **CONV1D_ARGS)(pi)
                            pi = Conv1D(x_shape[2], 1, 1,
                                        name=scope+'pi_conv_1', **CONV1D_ARGS)(pi)
                            x = Add(name=scope + 'add')([x, pi])

                x = Reshape((x_shape[1], x_shape[2], 1), name=scope+'resh_out')(x)

            with scope.name_scope('CNN'):
                n_hidden = 32
                u_blocks = 0
                min_dim = min(int(x.shape[1]), int(x.shape[2]))
                while min_dim > 4:
                    min_dim //= 2
                    u_blocks += 1
                u_blocks = min(u_blocks, 4)
                u_blocks = u_blocks * 2
                block_factors = range(1, (u_blocks // 2) + 1) + range(u_blocks // 2, 0, -1)
                n_stages = 1

                u_skips = []
                for k in range(n_stages):
                    with scope.name_scope('stage_%d' % k):
                        x = Conv2D(n_hidden, 1, 1, name=scope+'conv_in', **CONV2D_ARGS)(x)
                        pi = x
                        for i, factor in enumerate(block_factors):
                            with scope.name_scope('block_%d' % i):
                                n_filters = n_hidden * factor
                                if i < (u_blocks // 2):
                                    conv_func = Conv2D
                                    u_skips.append(pi)
                                else:
                                    conv_func = Conv2DTranspose

                                with scope.name_scope('pi'):
                                    pi = _conv_block(pi, n_filters, 2, 3, 2, conv_func)

                                if (u_blocks // 2) <= i < u_blocks:
                                    skip_pi = u_skips.pop()
                                    if skip_pi.shape[1] != pi.shape[1] or skip_pi.shape[2] != pi.shape[2]:
                                        pi = Cropping2D(((0, int(pi.shape[1] - skip_pi.shape[1])),
                                                        (0, int(pi.shape[2] - skip_pi.shape[2]))),
                                                         name=scope+'crop_pi')(pi)
                                    pi = Concatenate(name=scope+'cat_skip')([skip_pi, pi])
                                    with scope.name_scope('skip_pi'):
                                        pi = _conv_block(pi, n_filters, 2, 3, 1, conv_func)

                        x = Add(name=scope+'add')([x, pi])

        return x

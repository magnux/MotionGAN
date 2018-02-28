from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow.contrib.keras.api.keras.backend as K
from scipy.fftpack import idct
from scipy.linalg import pinv
from tensorflow.contrib.keras.api.keras.models import Model, Sequential
from tensorflow.contrib.keras.api.keras.layers import Input
from tensorflow.contrib.keras.api.keras.layers import Conv2DTranspose, Conv2D, \
    Dense, Activation, Lambda, Add, Concatenate, Permute, Reshape, Flatten, \
    Conv1D, Multiply
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.regularizers import l2
from layers.joints import UnfoldJoints, FoldJoints
from layers.normalization import InstanceNormalization
from collections import OrderedDict

CONV1D_ARGS = {'padding': 'same', 'kernel_regularizer': l2(5e-4)}
CONV2D_ARGS = {'padding': 'same', 'data_format': 'channels_last', 'kernel_regularizer': l2(5e-4)}

def _get_tensor(tensors, name):
    if isinstance(tensors, list):
        return next(obj for obj in tensors if name in obj.name)
    else:
        return tensors


def _conv_block(x, out_filters, bneck_factor, kernel_size, strides, i, j, net_name, conv_func=Conv2D):
    if 'generator' in net_name:
        x = InstanceNormalization(axis=-1, name='%s/block_%d/branch_%d/inorm_in' % (net_name, i, j))(x)
    x = Activation('relu', name='%s/block_%d/branch_%d/relu_in' % (net_name, i, j))(x)
    x = conv_func(filters=out_filters // bneck_factor, kernel_size=kernel_size, strides=1,
                  name='%s/block_%d/branch_%d/conv_in' % (net_name, i, j), **CONV2D_ARGS)(x)
    if 'generator' in net_name:
        x = InstanceNormalization(axis=-1, name='%s/block_%d/branch_%d/inorm_out' % (net_name, i, j))(x)
    x = Activation('relu', name='%s/block_%d/branch_%d/relu_out' % (net_name, i, j))(x)
    x = conv_func(filters=out_filters, kernel_size=kernel_size, strides=strides,
                  name='%s/block_%d/branch_%d/conv_out' % (net_name, i, j), **CONV2D_ARGS)(x)
    return x

def _edm(x):
    x1 = K.expand_dims(x, axis=1)
    x2 = K.expand_dims(x, axis=2)
    # epsilon needed in sqrt to avoid numerical issues
    return K.sqrt(K.sum(K.square(x1 - x2), axis=-1) + K.epsilon())


class _MotionGAN(object):
    def __init__(self, config):
        self.name = config.model_type + '_' + config.model_version
        self.data_set = config.data_set
        self.batch_size = config.batch_size
        self.num_actions = config.num_actions
        self.seq_len = config.pick_num if config.pick_num > 0 else (
                       config.crop_len if config.crop_len > 0 else None)
        self.njoints = config.njoints
        self.unfolded_joints = self.njoints

        self.dropout = config.dropout
        self.lambda_grads = config.lambda_grads
        self.gamma_grads = 1.0
        self.rec_scale = 0.1
        self.action_cond = config.action_cond
        self.action_scale_d = 10.0
        self.action_scale_g = 10.0
        self.latent_cond_dim = config.latent_cond_dim
        self.latent_scale_d = 1.0
        self.latent_scale_g = 1.0
        self.coherence_loss = config.coherence_loss
        self.coherence_scale = 0.1
        self.displacement_loss = config.displacement_loss
        self.displacement_scale = 0.1
        self.shape_loss = config.shape_loss
        self.shape_scale = 1.0
        self.smoothing_loss = config.smoothing_loss
        self.smoothing_scale = 1.0
        self.smoothing_basis = 3
        self.time_pres_emb = config.time_pres_emb
        self.unfold = config.unfold
        self.vae_pose_enc = config.vae_pose_enc
        self.vae_original_dim = self.njoints * 3
        self.vae_intermediate_dim = 32
        self.vae_latent_dim = 16
        self.vae_epsilon_std = 1.0
        self.vae_scale = 10.0
        # self.z_dim = config.z_dim

        # Placeholders for training phase
        self.place_holders = []
        if self.action_cond:
            true_label = K.placeholder(shape=(self.batch_size,), dtype='int32', name='true_label')
            self.place_holders.append(true_label)

        # Discriminator
        real_seq = Input(batch_shape=(self.batch_size, self.njoints, self.seq_len, 3),
                         name='real_seq', dtype='float32')
        self.disc_inputs = [real_seq]
        self.real_outputs = self._proc_disc_outputs(self.discriminator(real_seq))
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
        if self.vae_pose_enc:
            vae_epsilon = Input(batch_shape=(self.batch_size, self.seq_len, self.vae_latent_dim),
                                name='vae_epsilon', dtype='float32')
            self.gen_inputs.append(vae_epsilon)
        x = self._proc_gen_inputs(self.gen_inputs)
        self.gen_outputs = self._proc_gen_outputs(self.generator(x))
        self.gen_model = Model(self.gen_inputs,
                               self.gen_outputs,
                               name=self.name + '_generator')
        self.fake_outputs = self.disc_model(self.gen_outputs[0])

        # Losses
        self.wgan_losses, self.disc_losses, self.gen_losses = self._build_loss()

        disc_loss = 0.0
        for loss in self.disc_losses.values():
            disc_loss += loss

        gen_loss = 0.0
        for loss in self.gen_losses.values():
            gen_loss += loss


        # Custom train functions
        disc_optimizer = Adam(lr=config.learning_rate, beta_1=0., beta_2=0.9)
        disc_training_updates = disc_optimizer.get_updates(disc_loss, self.disc_model.trainable_weights)
        self.disc_train_f = K.function(self.disc_inputs + self.gen_inputs + self.place_holders,
                                       self.wgan_losses.values() + self.disc_losses.values(),
                                       disc_training_updates)
        self.disc_eval_f = K.function(self.disc_inputs + self.gen_inputs + self.place_holders,
                                      self.wgan_losses.values() + self.disc_losses.values())
        self.disc_model = self._pseudo_build_model(self.disc_model, disc_optimizer)

        gen_optimizer = Adam(lr=config.learning_rate, beta_1=0., beta_2=0.9)
        gen_training_updates = gen_optimizer.get_updates(gen_loss,
                               self.gen_model.trainable_weights)
        self.gen_train_f = K.function(self.gen_inputs + self.place_holders,
                                      self.gen_losses.values(),
                                      gen_training_updates)
        gen_f_outs = self.gen_losses.values()
        if self.vae_pose_enc:
            gen_f_outs.append(self.vae_z)
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
        if self.vae_pose_enc:
            keys.append('vae_z')
        keys.append('gen_outputs')
        losses_dict = OrderedDict(zip(keys, eval_outs))
        return losses_dict

    def update_lr(self, lr):
        K.set_value(self.disc_model.optimizer.lr, lr)
        K.set_value(self.gen_model.optimizer.lr, lr)

    def _build_loss(self):
        # Dicts to store the losses
        wgan_losses = OrderedDict()
        disc_losses = OrderedDict()
        gen_losses = OrderedDict()

        # WGAN Basic losses
        loss_real = K.mean(_get_tensor(self.real_outputs, 'score_out'), axis=-1)
        loss_fake = K.mean(_get_tensor(self.fake_outputs, 'score_out'), axis=-1)
        wgan_losses['loss_real'] = K.mean(loss_real)
        wgan_losses['loss_fake'] = K.mean(loss_fake)

        # Interpolates for GP
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        interpolates = (alpha * (self.disc_inputs[0])) + ((1 - alpha) * self.gen_outputs[0])

        # Gradient Penalty
        inter_out = _get_tensor(self.disc_model(interpolates), 'discriminator/score_out')
        grad_mixed = K.gradients(inter_out, [interpolates])[0]
        norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=(1, 2, 3)))
        grad_penalty = K.mean(K.square(norm_grad_mixed - self.gamma_grads) / (self.gamma_grads ** 2), axis=-1)

        # WGAN-GP losses
        disc_loss_wgan = loss_fake - loss_real + (self.lambda_grads * grad_penalty)
        disc_losses['disc_loss_wgan'] = K.mean(disc_loss_wgan)

        gen_loss_wgan = -loss_fake
        gen_losses['gen_loss_wgan'] = K.mean(gen_loss_wgan)

        # Regularization losses
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

        # Grabbing tensors
        real_seq = _get_tensor(self.gen_inputs, 'real_seq')
        seq_mask = _get_tensor(self.gen_inputs, 'seq_mask')
        gen_seq = self.gen_outputs[0]

        # Reconstruction loss
        loss_rec = K.sum(K.mean(K.square(real_seq - gen_seq) *
                                seq_mask, axis=-1), axis=(1, 2))
        gen_losses['gen_loss_rec'] = self.rec_scale * K.mean(loss_rec)

        if self.vae_pose_enc:
            vae_loss_dec = K.sum(K.mean(K.square((real_seq * seq_mask) - self.vae_dec_x), axis=-1), axis=(1, 2))
            gen_losses['vae_loss_dec'] = self.vae_scale * K.mean(vae_loss_dec)
            vae_loss_rec = K.sum(K.mean(K.abs(self.vae_z - self.vae_gen_z) *
                                        K.min(seq_mask, axis=1), axis=-1), axis=1)
            gen_losses['vae_loss_rec'] = self.vae_scale * K.mean(vae_loss_rec)
            vae_loss_kl = K.mean(- 0.5 * K.sum(1 + self.vae_z_log_var -
                                               K.square(self.vae_z_mean) -
                                               K.exp(self.vae_z_log_var), axis=-1))
            gen_losses['vae_loss_kl'] = self.vae_scale * K.mean(vae_loss_kl)

        # Conditional losses
        if self.action_cond:
            loss_class_real = K.mean(K.sparse_categorical_crossentropy(
                _get_tensor(self.place_holders, 'true_label'),
                _get_tensor(self.real_outputs, 'label_out'), True))
            loss_class_fake = K.mean(K.sparse_categorical_crossentropy(
                _get_tensor(self.place_holders, 'true_label'),
                _get_tensor(self.fake_outputs, 'label_out'), True))
            disc_losses['disc_loss_action'] = self.action_scale_d * (loss_class_real + loss_class_fake)
            gen_losses['gen_loss_action'] = self.action_scale_g * loss_class_fake
        if self.latent_cond_dim > 0:
            loss_latent = K.mean(K.square(_get_tensor(self.fake_outputs, 'latent_cond_out')
                                          - _get_tensor(self.gen_inputs, 'latent_cond_input')))
            disc_losses['disc_loss_latent'] = self.latent_scale_d * loss_latent
            gen_losses['gen_loss_latent'] = self.latent_scale_g * loss_latent
        if self.coherence_loss:
            exp_decay = 1.0 / np.exp(np.linspace(0.0, 2.0, self.seq_len, dtype='float32'))
            exp_decay = np.reshape(exp_decay, (1, 1, 1, self.seq_len))
            loss_coh = K.sum(K.mean(K.square(_edm(real_seq) - _edm(gen_seq)) * exp_decay, axis=-1), axis=(1, 2))
            gen_losses['gen_loss_coh'] = self.coherence_scale * K.mean(loss_coh)
        if self.displacement_loss:
            gen_seq_l = gen_seq[:, :, :-1, :]
            gen_seq_r = gen_seq[:, :, 1:, :]
            loss_disp = K.sum(K.mean(K.square(gen_seq_l - gen_seq_r), axis=-1), axis=(1, 2))
            gen_losses['gen_loss_disp'] = self.displacement_scale * K.mean(loss_disp)
        if self.shape_loss:
            mask = np.ones((self.njoints, self.njoints), dtype='float32')
            mask = np.triu(mask, 1) - np.triu(mask, 2)
            mask = np.reshape(mask, (1, self.njoints, self.njoints, 1))
            real_shape = K.mean(_edm(real_seq), axis=-1, keepdims=True) * mask
            gen_shape = _edm(gen_seq) * mask
            loss_shape = K.sum(K.mean(K.square(real_shape - gen_shape), axis=-1), axis=(1, 2))
            gen_losses['gen_loss_shape'] = self.shape_scale * K.mean(loss_shape)
        if self.smoothing_loss:
            Q = idct(np.eye(self.seq_len))[:self.smoothing_basis, :]
            Q_inv = pinv(Q)
            Qs = K.constant(np.matmul(Q_inv, Q))
            gen_seq_s = K.permute_dimensions(gen_seq, (0, 1, 3, 2))
            gen_seq_s = K.dot(gen_seq_s, Qs)
            gen_seq_s = K.permute_dimensions(gen_seq_s, (0, 1, 3, 2))
            loss_smooth = K.sum(K.mean(K.square(gen_seq_s - gen_seq), axis=-1), axis=(1, 2))
            gen_losses['gen_loss_smooth'] = self.smoothing_scale * K.mean(loss_smooth)

        return wgan_losses, disc_losses, gen_losses

    def _pseudo_build_model(self, model, optimizer):
        # This function mimics compilation to enable saving the model
        model.optimizer = optimizer
        model.sample_weight_mode = None
        model.loss = 'custom_loss'
        model.loss_weights = None
        model.metrics = None
        return model

    def _proc_disc_outputs(self, x):
        score_out = Dense(1, name='discriminator/score_out')(x)

        output_tensors = [score_out]
        if self.action_cond:
            label_out = Dense(self.num_actions, name='discriminator/label_out')(x)
            output_tensors.append(label_out)
        if self.latent_cond_dim > 0:
            latent_cond_out = Dense(self.latent_cond_dim, name='discriminator/latent_cond_out')(x)
            output_tensors.append(latent_cond_out)

        return output_tensors

    def _proc_gen_inputs(self, input_tensors):
        n_hidden = 32 if self.time_pres_emb else 128

        x = _get_tensor(input_tensors, 'real_seq')
        x_mask = _get_tensor(input_tensors, 'seq_mask')
        x = Multiply(name='generator/mask_mult')([x, x_mask])

        if self.unfold:
            x = UnfoldJoints(self.data_set)(x)
            self.unfolded_joints = int(x.shape[1])

        if self.vae_pose_enc:
            x = Permute((2, 1, 3))(x)  # time, joints, coords
            x = Reshape((self.seq_len, self.njoints * 3))(x)

            h = x
            for i in range(3):
                h = Conv1D(self.vae_intermediate_dim, 1, 1, activation='relu',
                           name='generator/pose_enc/vae_h_%d' % i, **CONV1D_ARGS)(h)
                
            self.vae_z_mean = Conv1D(self.vae_latent_dim, 1, 1, name='generator/pose_enc/vae_z_mean', **CONV1D_ARGS)(h)
            self.vae_z_log_var = Conv1D(self.vae_latent_dim, 1, 1, name='generator/pose_enc/vae_z_log_var', **CONV1D_ARGS)(h)

            # Had to discard this function because it breaks keras model.save()
            # def sampling(args):
            #     z_mean, z_log_var = args
            #     epsilon = K.random_normal(
            #         shape=(self.batch_size, self.seq_len, self.vae_latent_dim), mean=0.,
            #         stddev=self.vae_epsilon_std)
            #     return z_mean + K.exp(z_log_var / 2) * epsilon

            vae_epsilon = _get_tensor(input_tensors, 'vae_epsilon')
            self.vae_z = Lambda(lambda args: args[0] + (K.exp(args[1] / 2) * K.exp(args[2])),
                                output_shape=(self.vae_latent_dim,),
                                name='generator/pose_enc/vae_sampling')(
                             [self.vae_z_mean, self.vae_z_log_var, vae_epsilon])

            self.vae_decoder = Sequential()
            self.vae_decoder.add(Lambda(lambda x: x, input_shape=(self.seq_len, self.vae_latent_dim)))

            for i in range(3):
                self.vae_decoder.add(Conv1D(self.vae_intermediate_dim, 1, 1, activation='relu',
                                            name='generator/pose_enc/vae_dec_h_%d' % i, **CONV1D_ARGS))

            self.vae_decoder.add(Conv1D(self.vae_original_dim, 1, 1, name='generator/pose_enc/vae_dec_x', **CONV1D_ARGS))

            vae_dec_x = self.vae_decoder(self.vae_z)
            vae_dec_x = Reshape((self.seq_len, self.njoints, 3), name='generator/gen_out')(vae_dec_x)
            self.vae_dec_x = Permute((2, 1, 3))(vae_dec_x)

            x = Permute((2, 1))(self.vae_z)  # embedding, time
            x = Reshape((self.vae_latent_dim, self.seq_len, 1))(x)

            self.nblocks = 5

        else:
            strides = (2, 1) if self.time_pres_emb else 2
            i = 0
            while (x.shape[1] > 1 and self.time_pres_emb) or (i < 3):
                num_block = n_hidden * (((i + 1) // 2) + 1)
                shortcut = Conv2D(num_block, 1, strides,
                                  name='generator/seq_fex/block_%d/shortcut' % i, **CONV2D_ARGS)(x)
                pi = _conv_block(x, num_block, 8, 3, strides, i, 0, 'generator/seq_fex')
                x = Add(name='generator/seq_fex/block_%d/add' % i)([shortcut, pi])
                x = Activation('relu', name='generator/seq_fex/block_%d/relu_out' % i)(x)
                i += 1

            self.nblocks = i

            if not self.time_pres_emb:
                x = Lambda(lambda x: K.mean(x, axis=(1, 2)), name='generator/seq_fex/mean_pool')(x)

        x = [x]
        if self.latent_cond_dim > 0:
            x_lat = _get_tensor(input_tensors, 'latent_cond_input')
            x.append(x_lat)

        if len(x) > 1:
            x = Concatenate(name='generator/cat_in')(x)
        else:
            x = x[0]

        return x

    def _proc_gen_outputs(self, x):

        if self.vae_pose_enc:
            x = Conv2D(1, 3, 1, name='generator/vae_merge', **CONV2D_ARGS)(x)
            x = Reshape((self.vae_latent_dim, self.seq_len))(x)
            self.vae_gen_z = Permute((2, 1))(x)

            vae_gen_x = self.vae_decoder(self.vae_gen_z)

            x = Reshape((self.seq_len, self.njoints, 3), name='generator/gen_out')(vae_gen_x)
            x = Permute((2, 1, 3))(x)  # joints, time, coords
        else:
            x = Permute((3, 2, 1))(x)  # filters, time, joints
            x = Conv2D(self.unfolded_joints, 3, 1, name='generator/joint_reshape', **CONV2D_ARGS)(x)
            x = Permute((1, 3, 2))(x)  # filters, joints, time
            x = Conv2D(self.seq_len, 3, 1, name='generator/time_reshape', **CONV2D_ARGS)(x)
            x = Permute((2, 3, 1))(x)  # joints, time, filters
            x = Conv2D(3, 3, 1, name='generator/gen_out', **CONV2D_ARGS)(x)

        if self.unfold:
            x = FoldJoints(self.data_set)(x)

        output_tensors = [x]

        return output_tensors


class MotionGANV1(_MotionGAN):
    # ResNet

    def discriminator(self, x):
        n_hidden = 64
        block_factors = [1, 1, 2, 2]
        block_strides = [2, 2, 1, 1]

        x = Conv2D(n_hidden * block_factors[0], 3, 1, name='discriminator/conv_in', **CONV2D_ARGS)(x)
        for i, factor in enumerate(block_factors):
            n_filters = n_hidden * factor
            shortcut = Conv2D(n_filters, block_strides[i], block_strides[i],
                              name='discriminator/block_%d/shortcut' % i, **CONV2D_ARGS)(x)
            pi = _conv_block(x, n_filters, 1, 3, block_strides[i], i, 0, 'discriminator')

            x = Add(name='discriminator/block_%d/add' % i)([shortcut, pi])

        x = Activation('relu', name='discriminator/relu_out')(x)
        x = Lambda(lambda x: K.mean(x, axis=(1, 2)), name='discriminator/mean_pool')(x)

        return x

    def generator(self, x):
        n_hidden = 16
        block_factors = range(1, self.nblocks + 1)
        block_strides = [2] * self.nblocks

        if not (self.time_pres_emb or self.vae_pose_enc):
            x = Dense(4 * 4 * n_hidden * block_factors[0], name='generator/dense_in')(x)
            x = Reshape((4, 4, n_hidden * block_factors[0]), name='generator/reshape_in')(x)

        for i, factor in enumerate(block_factors):
            n_filters = n_hidden * factor
            strides = block_strides[i]
            if self.time_pres_emb:
                strides = (block_strides[i], 1)
            elif self.vae_pose_enc:
                strides = 1
            shortcut = Conv2DTranspose(n_filters, strides, strides,
                                       name='generator/block_%d/shortcut' % i, **CONV2D_ARGS)(x)
            pi = _conv_block(x, n_filters, 1, 3, strides, i, 0,
                             'generator', Conv2DTranspose)

            x = Add(name='generator/block_%d/add' % i)([shortcut, pi])

        x = InstanceNormalization(axis=-1, name='generator/inorm_out')(x)
        x = Activation('relu', name='generator/relu_out')(x)

        return x


class MotionGANV2(_MotionGAN):
    # Gated ResNet

    def discriminator(self, x):
        n_hidden = 64
        block_factors = [1, 1, 2, 2]
        block_strides = [2, 2, 1, 1]

        x = Conv2D(n_hidden * block_factors[0], 3, 1, name='discriminator/conv_in', **CONV2D_ARGS)(x)
        for i, factor in enumerate(block_factors):
            n_filters = n_hidden * factor
            shortcut = Conv2D(n_filters, block_strides[i], block_strides[i],
                              name='discriminator/block_%d/shortcut' % i, **CONV2D_ARGS)(x)

            pi = _conv_block(x, n_filters, 1, 3, block_strides[i], i, 0, 'discriminator')
            gamma = _conv_block(x, n_filters, 4, 3, block_strides[i], i, 1, 'discriminator')
            gamma = Activation('sigmoid', name='discriminator/block_%d/sigmoid' % i)(gamma)

            # tau = 1 - gamma
            tau = Lambda(lambda arg: 1 - arg, name='discriminator/block_%d/tau' % i)(gamma)

            # x = (pi * tau) + (shortcut * gamma)
            x = Lambda(lambda args: (args[0] * args[1]) + (args[2] * args[3]),
                       name='discriminator/block_%d/out_x' % i)([pi, tau, shortcut, gamma])

        x = Activation('relu', name='discriminator/relu_out')(x)
        x = Lambda(lambda x: K.mean(x, axis=(1, 2)), name='discriminator/mean_pool')(x)

        return x

    def generator(self, x):
        n_hidden = 16
        block_factors = range(1, self.nblocks + 1)
        block_strides = [2] * self.nblocks

        # For condition injecting
        # z = x

        if not (self.time_pres_emb or self.vae_pose_enc):
            for i in range(2):
                if i > 0:
                    x = InstanceNormalization(axis=-1, name='generator/dense_block%d/bn' % i)(x)
                    x = Activation('relu', name='generator/dense_block%d/relu' % i)(x)
                x = Dense(n_hidden * 4, name='generator/dense_block%d/dense' % i)(x)

            x = InstanceNormalization(axis=-1, name='generator/inorm_conv_in')(x)
            x = Activation('relu', name='generator/relu_conv_in')(x)
            x = Dense(4 * 4 * n_hidden * block_factors[0], name='generator/dense_conv_in')(x)
            x = Reshape((4, 4, n_hidden * block_factors[0]), name='generator/reshape_conv_in')(x)

        for i, factor in enumerate(block_factors):
            n_filters = n_hidden * factor
            strides = block_strides[i]
            if self.time_pres_emb:
                strides = (block_strides[i], 1)
            elif self.vae_pose_enc:
                strides = 1
            shortcut = Conv2DTranspose(n_filters, strides, strides,
                                       name='generator/block_%d/shortcut' % i, **CONV2D_ARGS)(x)

            pi = _conv_block(x, n_filters, 2, 3, strides, i, 0, 'generator', Conv2DTranspose)

            # For condition injecting
            # squeeze_kernel = (x.shape[1], x.shape[2])
            # gamma = _conv_block(x, n_filters, 8, squeeze_kernel, squeeze_kernel, i, 1, 'generator')
            # gamma = Flatten(name='generator/block_%d/gamma_flatten' % i)(gamma)
            # gamma = Concatenate(name='generator/block_%d/cond_concat' % i)([gamma, z])
            # gamma = InstanceNormalization(axis=-1, name='generator/block_%d/cond_bn' % i)(gamma)
            # gamma = Activation('relu', name='generator/block_%d/cond_relu' % i)(gamma)
            # gamma = Dense(n_filters, name='generator/block_%d/cond_dense' % i)(gamma)
            # gamma = Reshape((1, 1, n_filters), name='generator/block_%d/gamma_reshape' % i)(gamma)

            gamma = _conv_block(x, n_filters, 4, 3, strides, i, 1, 'generator', Conv2DTranspose)
            gamma = Activation('sigmoid', name='generator/block_%d/gamma_sigmoid' % i)(gamma)

            # tau = 1 - gamma
            tau = Lambda(lambda arg: 1 - arg, name='generator/block_%d/tau' % i)(gamma)

            # x = (pi * tau) + (shortcut * gamma)
            x = Lambda(lambda args: (args[0] * args[1]) + (args[2] * args[3]),
                       name='generator/block_%d/out_x' % i)([pi, tau, shortcut, gamma])

        x = InstanceNormalization(axis=-1, name='generator/inorm_out')(x)
        x = Activation('relu', name='generator/relu_out')(x)

        return x


def _preact_dense(x, n_units, i, j):
    x = InstanceNormalization(axis=-1, name='generator/block_%d/inorm_%d' % (i, j))(x)
    x = Activation('relu', name='generator/block_%d/relu_%d' % (i, j))(x)
    x = Dense(n_units, name='generator/block_%d/dense_%d' % (i, j), activation='relu')(x)
    return x


class MotionGANV3(_MotionGAN):
    # Simple dense ResNet

    def discriminator(self, x):

        x = Reshape((self.njoints * self.seq_len * 3, ), name='discriminator/reshape_in')(x)
        x = Dense(1024, name='discriminator/block_0/dense_0', activation='relu')(x)
        for i in range(1, 5):
            pi = Dense(1024, name='discriminator/block_%d/dense_0' % i, activation='relu')(x)
            pi = Dense(1024, name='discriminator/block_%d/dense_1' % i, activation='relu')(pi)

            x = Add(name='discriminator/block_%d/add' % i)([x, pi])

        return x

    def generator(self, x):

        x = Flatten(name='generator/flatten_in')(x)
        x = _preact_dense(x, 1024, 0, 0)
        for i in range(1, 5):
            pi = _preact_dense(x, 1024, i, 0)
            pi = _preact_dense(pi, 1024, i, 1)

            x = Add(name='generator/block_%d/add' % i)([x, pi])

        x = Dense((self.unfolded_joints * (self.seq_len // 2) * 3), name='generator/dense_out', activation='relu')(x)
        x = Reshape((self.unfolded_joints, self.seq_len // 2, 3), name='generator/reshape_out')(x)

        return x
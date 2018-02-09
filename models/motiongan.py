from __future__ import absolute_import, division, print_function
import numpy as np
from tensorflow.contrib.keras.api.keras.backend import sum, mean, \
    square, sqrt, random_uniform, gradients, placeholder, set_value, \
    sparse_categorical_crossentropy, cast_to_floatx, expand_dims, max
from tensorflow.contrib.keras.api.keras.backend import function as K_function
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Input
from tensorflow.contrib.keras.api.keras.layers import Conv2DTranspose, \
    Conv2D, Dense, Dropout, Flatten, LeakyReLU, Reshape, Activation,\
    BatchNormalization, Lambda, Add, Concatenate, Cropping2D, Permute
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.initializers import RandomNormal
from tensorflow.contrib.keras.api.keras.regularizers import l2


def _get_tensor(tensors, name):
    if isinstance(tensors, list):
        return next(obj for obj in tensors if name in obj.name)
    else:
        return tensors


def _conv_block(x, out_filters, bneck_factor, kernel_size, strides, i, j, net_name, training=None, conv_func=Conv2D):
    conv_args = {'padding': 'same', 'data_format': 'channels_last', 'kernel_regularizer': l2(5e-4)}
    if 'generator' in net_name:
        x = BatchNormalization(name='%s/block_%d/branch_%d/bn_in' % (net_name, i, j))(x, training=training)
    x = Activation('relu', name='%s/block_%d/branch_%d/relu_in' % (net_name, i, j))(x)
    x = conv_func(filters=out_filters // bneck_factor, kernel_size=kernel_size, strides=1,
                  name='%s/block_%d/branch_%d/conv_in' % (net_name, i, j), **conv_args)(x)
    if 'generator' in net_name:
        x = BatchNormalization(name='%s/block_%d/branch_%d/bn_out' % (net_name, i, j))(x, training=training)
    x = Activation('relu', name='%s/block_%d/branch_%d/relu_out' % (net_name, i, j))(x)
    x = conv_func(filters=out_filters, kernel_size=kernel_size, strides=strides,
                  name='%s/block_%d/branch_%d/conv_out' % (net_name, i, j), **conv_args)(x)
    return x


def _shape_seq_out(x, njoints, seq_len):
    conv_args = {'padding': 'same', 'data_format': 'channels_last', 'kernel_regularizer': l2(5e-4)}
    x = Permute((3, 2, 1))(x)  # filters, time, joints
    x = Conv2D(njoints, 3, 1, name='generator/joint_reshape', **conv_args)(x)
    x = Permute((1, 3, 2))(x)  # filters, joints, time
    x = Conv2D(seq_len // 2, 3, 1, name='generator/time_reshape', **conv_args)(x)
    x = Permute((2, 3, 1))(x)  # joints, time, filters
    x = Conv2D(3, 3, 1, name='generator/conv_out', **conv_args)(x)
    return x


def _edm(x):
    x1 = expand_dims(x, axis=1)
    x2 = expand_dims(x, axis=2)
    # epsilon needed in sqrt to avoid numerical issues
    return sqrt(sum(square(x1 - x2), axis=-1) + 1e-8)


class _MotionGAN(object):
    def __init__(self, config):
        self.name = config.model_type + '_' + config.model_version

        self.batch_size = config.batch_size
        # self.z_dim = config.z_dim
        self.num_actions = config.num_actions
        self.dropout = config.dropout
        self.lambda_grads = config.lambda_grads
        self.gamma_grads = 1e0
        self.action_cond = config.action_cond
        self.action_scale_d = 1e0
        self.action_scale_g = 1e-1
        self.latent_cond_dim = config.latent_cond_dim
        self.latent_scale_d = 1e0
        self.latent_scale_g = 1e-1
        self.coherence_loss = config.coherence_loss
        self.coherence_scale = 1e-1
        self.displacement_loss = config.displacement_loss
        self.displacement_scale = 1e-2
        self.shape_loss = config.shape_loss
        self.shape_scale = 1e-1
        self.time_pres_emb = config.time_pres_emb

        self.seq_len = config.pick_num if config.pick_num > 0 else (
                       config.crop_len if config.crop_len > 0 else None)
        self.njoints = config.njoints

        # Placeholders for training phase
        self.disc_training = placeholder(shape=(), dtype='bool', name='disc_training')
        self.place_holders = [self.disc_training]
        if self.action_cond:
            true_label = placeholder(shape=(self.batch_size,), dtype='int32', name='true_label')
            self.place_holders.append(true_label)

        self.gen_training = (self.disc_training == False)

        # Discriminator
        real_seq = Input(batch_shape=(self.batch_size, config.njoints, self.seq_len, 3),
                         name='real_seq', dtype='float32')
        self.disc_inputs = [real_seq]
        self.real_outputs = self._proc_disc_outputs(self.discriminator(real_seq))
        self.disc_model = Model(self.disc_inputs,
                                self.real_outputs,
                                name=self.name + '_discriminator')

        # Generator
        self.gen_inputs = [real_seq]
        if self.latent_cond_dim > 0:
            latent_cond_input = Input(batch_shape=(self.batch_size, self.latent_cond_dim),
                                      name='latent_cond_input', dtype='float32')
            self.gen_inputs.append(latent_cond_input)
        x, seq_head = self._prep_gen_inputs(self.gen_inputs)
        gen_seq = Concatenate(axis=2, name='gen_seq')([seq_head, self.generator(x)])
        self.gen_outputs = [gen_seq]
        self.gen_model = Model(self.gen_inputs,
                               self.gen_outputs,
                               name=self.name + '_generator')
        self.fake_outputs = self.disc_model(self.gen_outputs[0])

        # Losses
        loss_real, loss_fake, disc_loss, gen_loss = self._build_wgan_gp_loss()


        # Custom train functions
        disc_optimizer = Adam(lr=config.learning_rate, beta_1=0., beta_2=0.9)
        disc_training_updates = disc_optimizer.get_updates(disc_loss,
                                self.disc_model.trainable_weights)
        self.disc_train = K_function(self.disc_inputs + self.gen_inputs + self.place_holders,
                                     [loss_real, loss_fake],
                                     disc_training_updates)
        self.disc_model = self._pseudo_build_model(self.disc_model, disc_optimizer)

        gen_optimizer = Adam(lr=config.learning_rate, beta_1=0., beta_2=0.9)
        gen_training_updates = gen_optimizer.get_updates(gen_loss,
                               self.gen_model.trainable_weights)
        self.gen_train = K_function(self.gen_inputs + self.place_holders,
                                    [gen_loss],
                                    gen_training_updates)
        self.gen_model = self._pseudo_build_model(self.gen_model, gen_optimizer)

        # GAN, complete model
        self.gan_model = Model(self.gen_inputs,
                               self.disc_model(self.gen_model(self.gen_inputs)),
                               name=self.name + '_gan')

    def update_lr(self, lr):
        set_value(self.disc_model.optimizer.lr, lr)
        set_value(self.gen_model.optimizer.lr, lr)

    def _build_wgan_gp_loss(self):
        # Basic losses
        loss_real = mean(_get_tensor(self.real_outputs, 'score_out'))
        loss_fake = mean(_get_tensor(self.fake_outputs, 'score_out'))

        # Interpolates
        alpha = random_uniform((self.batch_size, 1, 1, 1))
        interpolates = (alpha * (self.disc_inputs[0])) + ((1 - alpha) * self.gen_outputs[0])

        # Gradient penalty
        inter_out = _get_tensor(self.disc_model(interpolates), 'discriminator/score_out')
        grad_mixed = gradients(inter_out, [interpolates])[0]
        norm_grad_mixed = sqrt(sum(square(grad_mixed), axis=(1, 2, 3)))
        grad_penalty = mean(square(norm_grad_mixed - self.gamma_grads) / (self.gamma_grads ** 2))

        # WGAN-GP losses
        disc_loss = loss_fake - loss_real + (self.lambda_grads * grad_penalty)
        gen_loss = -loss_fake

        # Regularization losses
        for reg_loss in set(self.disc_model.losses):
            disc_loss += reg_loss
        for reg_loss in set(self.gen_model.losses):
            gen_loss += reg_loss

        # Conditional losses
        if self.action_cond:
            loss_class_real = mean(sparse_categorical_crossentropy(
                _get_tensor(self.place_holders, 'true_label'),
                _get_tensor(self.real_outputs, 'label_out'), True))
            loss_class_fake = mean(sparse_categorical_crossentropy(
                _get_tensor(self.place_holders, 'true_label'),
                _get_tensor(self.fake_outputs, 'label_out'), True))
            disc_loss += (self.action_scale_d * (loss_class_real + loss_class_fake))
            gen_loss += (self.action_scale_g * loss_class_fake)
        if self.latent_cond_dim > 0:
            loss_latent = mean(square(_get_tensor(self.fake_outputs, 'latent_cond_out')
                                      - _get_tensor(self.gen_inputs, 'latent_cond_input')))
            disc_loss += (self.latent_scale_d * loss_latent)
            gen_loss += (self.latent_scale_g * loss_latent)
        if self.coherence_loss:
            seq_tail = self.gen_inputs[0][:, :, self.seq_len // 2:, :]
            gen_tail = self.gen_outputs[0][:, :, self.seq_len // 2:, :]
            exp_decay = 1.0 / np.exp(np.linspace(0.0, 2.0, self.seq_len // 2, dtype='float32'))
            exp_decay = np.reshape(exp_decay, (1, 1, 1, self.seq_len // 2))
            loss_coh = mean(square(_edm(seq_tail) - _edm(gen_tail)) * exp_decay)
            gen_loss += (self.coherence_scale * loss_coh)
        if self.displacement_loss:
            gen_tail = self.gen_outputs[0][:, :, self.seq_len // 2:, :]
            gen_tail_s = self.gen_outputs[0][:, :, (self.seq_len // 2)-1:-1, :]
            loss_disp = mean(square(gen_tail - gen_tail_s))
            gen_loss += (self.displacement_scale * loss_disp)
        if self.shape_loss:
            mask = np.ones((self.njoints, self.njoints), dtype='float32')
            mask = np.triu(mask, 1) - np.triu(mask, 2)
            mask = np.reshape(mask, (1, self.njoints, self.njoints, 1))
            real_shape = mean(_edm(self.gen_inputs[0]), axis=-1, keepdims=True) * mask
            gen_tail = self.gen_outputs[0][:, :, self.seq_len // 2:, :]
            gen_shape = _edm(gen_tail) * mask
            loss_shape = mean(square(real_shape - gen_shape))
            gen_loss += (self.shape_scale * loss_shape)

        return loss_real, loss_fake, disc_loss, gen_loss

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

    def _prep_gen_inputs(self, input_tensors):
        conv_args = {'padding': 'same', 'data_format': 'channels_last', 'kernel_regularizer': l2(5e-4)}
        n_hidden = 32 if self.time_pres_emb else 128
        seq_head = _get_tensor(input_tensors, 'real_seq')
        seq_head = Cropping2D(((0, 0), (0, self.seq_len // 2)), name='seq_head')(seq_head)
        x = seq_head

        strides = (2, 1) if self.time_pres_emb else 2
        for i in range(3):
            num_block = n_hidden * (((i + 1) // 2) + 1)
            shortcut = Conv2D(num_block, 1, strides,
                              name='generator/seq_fex/block_%d/shortcut' % i, **conv_args)(x)
            pi = _conv_block(x, num_block, 8, 3, strides, i, 0, 'generator/seq_fex', self.gen_training)
            x = Add(name='generator/seq_fex/block_%d/add' % i)([shortcut, pi])
            x = Activation('relu', name='generator/seq_fex/block_%d/relu_out' % i)(x)

        if not self.time_pres_emb:
            x = Lambda(lambda x: mean(x, axis=(1, 2)), name='generator/seq_fex/mean_pool')(x)
        x = [x]
        if self.latent_cond_dim > 0:
            x_lat = _get_tensor(input_tensors, 'latent_cond_input')
            x.append(x_lat)

        if len(x) > 1:
            x = Concatenate(name='generator/cat_in')(x)
        else:
            x = x[0]

        return x, seq_head


class MotionGANV1(_MotionGAN):

    def discriminator(self, x):
        # ResNet discriminator
        conv_args = {'padding': 'same', 'data_format': 'channels_last', 'kernel_regularizer': l2(5e-4)}
        n_hidden = 64
        block_factors = [2, 2, 2, 2]
        block_strides = [2, 2, 1, 1]

        x = Conv2D(n_hidden * block_factors[0], 3, 1, name='discriminator/conv_in', **conv_args)(x)
        for i, factor in enumerate(block_factors):
            n_filters = n_hidden * factor
            shortcut = Conv2D(n_filters, block_strides[i], block_strides[i],
                              name='discriminator/block_%d/shortcut' % i, **conv_args)(x)
            pi = _conv_block(x, n_filters, 1, 3, block_strides[i], i, 0, 'discriminator')

            x = Add(name='discriminator/block_%d/add' % i)([shortcut, pi])

        x = Activation('relu', name='discriminator/relu_out')(x)
        x = Lambda(lambda x: mean(x, axis=(1, 2)), name='discriminator/mean_pool')(x)

        return x

    def generator(self, x):
        # ResNet generator
        conv_args = {'padding': 'same', 'data_format': 'channels_last', 'kernel_regularizer': l2(5e-4)}
        n_hidden = 64
        block_factors = [2, 2, 2]
        block_strides = [2, 2, 2]

        if not self.time_pres_emb:
            x = Dense(4 * 4 * n_hidden * block_factors[0], name='generator/dense_in')(x)
            x = Reshape((4, 4, n_hidden * block_factors[0]), name='generator/reshape_in')(x)

        for i, factor in enumerate(block_factors):
            n_filters = n_hidden * factor
            strides = (block_strides[i], 1) if self.time_pres_emb else block_strides[i]
            shortcut = Conv2DTranspose(n_filters, strides, strides,
                                       name='generator/block_%d/shortcut' % i, **conv_args)(x)
            pi = _conv_block(x, n_filters, 1, 3, strides, i, 0,
                             'generator', self.gen_training, Conv2DTranspose)

            x = Add(name='generator/block_%d/add' % i)([shortcut, pi])

        x = BatchNormalization(name='generator/bn_out')(x, training=self.gen_training)
        x = Activation('relu', name='generator/relu_out')(x)

        x = _shape_seq_out(x, self.njoints, self.seq_len)

        return x


class MotionGANV2(_MotionGAN):

    def discriminator(self, x):
        # Gated ResNet discriminator
        conv_args = {'padding': 'same', 'data_format': 'channels_last', 'kernel_regularizer': l2(5e-4)}
        n_hidden = 64
        block_factors = [2, 2, 2, 2]
        block_strides = [2, 2, 1, 1]

        x = Conv2D(n_hidden * block_factors[0], 3, 1, name='discriminator/conv_in', **conv_args)(x)
        for i, factor in enumerate(block_factors):
            n_filters = n_hidden * factor
            shortcut = Conv2D(n_filters, block_strides[i], block_strides[i],
                              name='discriminator/block_%d/shortcut' % i, **conv_args)(x)

            pi = _conv_block(x, n_filters, 1, 3, block_strides[i], i, 0, 'discriminator')
            gamma = _conv_block(x, n_filters, 4, 3, block_strides[i], i, 1, 'discriminator')
            gamma = Activation('sigmoid', name='discriminator/block_%d/sigmoid' % i)(gamma)

            # tau = 1 - gamma
            tau = Lambda(lambda arg: 1 - arg, name='discriminator/block_%d/tau' % i)(gamma)

            # x = (pi * tau) + (shortcut * gamma)
            x = Lambda(lambda args: (args[0] * args[1]) + (args[2] * args[3]),
                       name='discriminator/block_%d/out_x' % i)([pi, tau, shortcut, gamma])

        x = Activation('relu', name='discriminator/relu_out')(x)
        x = Lambda(lambda x: mean(x, axis=(1, 2)), name='discriminator/mean_pool')(x)

        return x

    def generator(self, x):
        # Gated ResNet generator
        conv_args = {'padding': 'same', 'data_format': 'channels_last', 'kernel_regularizer': l2(5e-4)}
        n_hidden = 64
        block_factors = [2, 2, 2]
        block_strides = [2, 2, 2]

        # For condition injecting
        # z = x

        if not self.time_pres_emb:
            for i in range(2):
                if i > 0:
                    x = BatchNormalization(name='generator/dense_block%d/bn' % i)(x, training=self.gen_training)
                    x = Activation('relu', name='generator/dense_block%d/relu' % i)(x)
                x = Dense(n_hidden * 4, name='generator/dense_block%d/dense' % i)(x)

            x = BatchNormalization(name='generator/bn_conv_in')(x, training=self.gen_training)
            x = Activation('relu', name='generator/relu_conv_in')(x)
            x = Dense(4 * 4 * n_hidden * block_factors[0], name='generator/dense_conv_in')(x)
            x = Reshape((4, 4, n_hidden * block_factors[0]), name='generator/reshape_conv_in')(x)

        for i, factor in enumerate(block_factors):
            n_filters = n_hidden * factor
            strides = (block_strides[i], 1) if self.time_pres_emb else block_strides[i]
            shortcut = Conv2DTranspose(n_filters, strides, strides,
                                       name='generator/block_%d/shortcut' % i, **conv_args)(x)

            pi = _conv_block(x, n_filters, 2, 3, strides, i, 0,
                             'generator', self.gen_training, Conv2DTranspose)
            # For condition injecting
            # squeeze_kernel = (x.shape[1], x.shape[2])
            # gamma = _conv_block(x, n_filters, 8, squeeze_kernel, squeeze_kernel, i, 1,
            #                     'generator', self.gen_training)
            # gamma = Flatten(name='generator/block_%d/gamma_flatten' % i)(gamma)
            # gamma = Concatenate(name='generator/block_%d/cond_concat' % i)([gamma, z])
            # gamma = BatchNormalization(name='generator/block_%d/cond_bn' % i)(gamma, training=self.gen_training)
            # gamma = Activation('relu', name='generator/block_%d/cond_relu' % i)(gamma)
            # gamma = Dense(n_filters, name='generator/block_%d/cond_dense' % i)(gamma)
            gamma = _conv_block(x, n_filters, 4, 3, strides, i, 1,
                                'generator', self.gen_training, Conv2DTranspose)
            gamma = Activation('sigmoid', name='generator/block_%d/gamma_sigmoid' % i)(gamma)
            gamma = Reshape((1, 1, n_filters), name='generator/block_%d/gamma_reshape' % i)(gamma)

            # tau = 1 - gamma
            tau = Lambda(lambda arg: 1 - arg, name='generator/block_%d/tau' % i)(gamma)

            # x = (pi * tau) + (shortcut * gamma)
            x = Lambda(lambda args: (args[0] * args[1]) + (args[2] * args[3]),
                       name='generator/block_%d/out_x' % i)([pi, tau, shortcut, gamma])

        x = BatchNormalization(name='generator/bn_out')(x, training=self.gen_training)
        x = Activation('relu', name='generator/relu_out')(x)

        x = _shape_seq_out(x, self.njoints, self.seq_len)

        return x



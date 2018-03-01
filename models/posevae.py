from __future__ import absolute_import, division, print_function
import tensorflow.contrib.keras.api.keras.backend as K
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Input
from tensorflow.contrib.keras.api.keras.layers import \
    Lambda, Add, Permute, Reshape, Conv1D, Multiply
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.regularizers import l2
from tensorflow.contrib.keras.api.keras.initializers import lecun_normal

CONV1D_ARGS = {'padding': 'same', 'kernel_regularizer': l2(5e-4), 'kernel_initializer': lecun_normal()}


class _PoseVAE(object):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.num_actions = config.num_actions
        self.seq_len = config.pick_num if config.pick_num > 0 else (
            config.crop_len if config.crop_len > 0 else None)
        self.njoints = config.njoints

        self.vae_original_dim = self.njoints * 3
        self.vae_intermediate_dim = self.vae_original_dim // 2
        # We are only expecting half of the latent features to be activated
        self.vae_latent_dim = self.vae_original_dim // 2
        self.vae_epsilon_std = 1.0

        pose_input = Input(batch_shape=(self.batch_size, self.njoints, self.seq_len, 3), dtype='float32')
        vae_epsilon = Input(batch_shape=(self.batch_size, self.seq_len, self.vae_latent_dim), dtype='float32')
        embedded_input = Input(shape=(self.seq_len, self.vae_latent_dim), dtype='float32')

        encoder_inputs = [pose_input, vae_epsilon]
        encoder_outputs = self.encoder(encoder_inputs)
        z_mean, z_log_var, z = encoder_outputs
        self.encoder = Model(encoder_inputs, encoder_outputs)

        decoder_inputs = embedded_input
        decoder_outputs = self.decoder(decoder_inputs)
        self.decoder = Model(decoder_inputs, decoder_outputs)

        dec_pose = self.decoder(z)
        self.autoencoder = Model(encoder_inputs, dec_pose)

        def vae_loss(y_true, y_pred):
            vae_loss_dec = K.sum(K.mean(K.square(y_true - y_pred), axis=-1), axis=(1, 2))
            loss = K.mean(vae_loss_dec)
            vae_loss_kl = K.sum(- 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1), axis=1)
            loss += K.mean(vae_loss_kl)
            return loss

        vae_optimizer = Adam(lr=config.learning_rate)
        self.autoencoder.compile(vae_optimizer, vae_loss, metrics=['mean_squared_error'])


class PoseVAEV1(_PoseVAE):
    
    def encoder(self, encoder_inputs):
        pose_input, vae_epsilon = encoder_inputs

        x = Permute((2, 1, 3), name='pose_vae/perm_in')(pose_input)  # time, joints, coords
        x = Reshape((self.seq_len, self.njoints * 3), name='pose_vae/resh_in')(x)

        h = Conv1D(self.vae_intermediate_dim, 1, 1,
                   name='pose_vae/enc_in', **CONV1D_ARGS)(x)
        for i in range(3):
            pi = Conv1D(self.vae_intermediate_dim, 1, 1, activation='relu',
                        name='pose_vae/enc_%d_0' % i, **CONV1D_ARGS)(h)
            pi = Conv1D(self.vae_intermediate_dim, 1, 1, activation='relu',
                        name='pose_vae/enc_%d_1' % i, **CONV1D_ARGS)(pi)
            tau = Conv1D(self.vae_intermediate_dim, 1, 1, activation='sigmoid',
                         name='pose_vae/enc_%d_tau' % i, **CONV1D_ARGS)(h)
            h = Lambda(lambda args: (args[0] * (1 - args[2])) + (args[1] * args[2]),
                       name='pose_vae/enc_%d_attention' % i)([h, pi, tau])

        z_mean = Conv1D(self.vae_latent_dim, 1, 1, name='pose_vae/z_mean', **CONV1D_ARGS)(h)
        z_log_var = Conv1D(self.vae_latent_dim, 1, 1, name='pose_vae/z_log_var', **CONV1D_ARGS)(h)
        z_attention = Conv1D(self.vae_latent_dim, 1, 1, activation='sigmoid',
                             name='pose_vae/z_attention', **CONV1D_ARGS)(h)

        # Had to discard this function because it breaks keras model.save()
        # def sampling(args):
        #     z_mean, z_log_var = args
        #     epsilon = K.random_normal(
        #         shape=(self.batch_size, self.seq_len, self.vae_latent_dim), mean=0.,
        #         stddev=self.vae_epsilon_std)
        #     return z_mean + K.exp(z_log_var / 2) * epsilon

        z = Lambda(lambda args: args[0] + (K.exp(args[1] / 2) * K.exp(args[2])),
                   output_shape=(self.vae_latent_dim,),
                   name='pose_vae/vae_sampling')([z_mean, z_log_var, vae_epsilon])

        z = Multiply(name='pose_vae/vae_attention')([z, z_attention])

        return z_mean, z_log_var, z
        
    def decoder(self, embedded_input):
        dec_h = Conv1D(self.vae_intermediate_dim, 1, 1,
                           name='pose_vae/dec_in', **CONV1D_ARGS)(embedded_input)
        for i in range(3):
            pi = Conv1D(self.vae_intermediate_dim, 1, 1, activation='relu',
                        name='pose_vae/dec_%d_0' % i, **CONV1D_ARGS)(dec_h)
            pi = Conv1D(self.vae_intermediate_dim, 1, 1, activation='relu',
                        name='pose_vae/dec_%d_1' % i, **CONV1D_ARGS)(pi)
            tau = Conv1D(self.vae_intermediate_dim, 1, 1, activation='sigmoid',
                         name='pose_vae/dec_%d_tau' % i, **CONV1D_ARGS)(dec_h)
            dec_h = Lambda(lambda args: (args[0] * (1 - args[2])) + (args[1] * args[2]),
                           name='pose_vae/dec_%d_attention' % i)([dec_h, pi, tau])

        dec_x = Conv1D(self.vae_original_dim, 1, 1, name='pose_vae/dec_out', **CONV1D_ARGS)(dec_h)

        dec_x = Reshape((self.seq_len, self.njoints, 3), name='pose_vae/resh_out')(dec_x)
        dec_x = Permute((2, 1, 3), name='pose_vae/perm_out')(dec_x)

        return dec_x

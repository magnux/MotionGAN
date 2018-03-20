from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow.contrib.keras.api.keras.backend as K
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Input
from tensorflow.contrib.keras.api.keras.layers import Conv2D, \
    Dense, Activation, Lambda, Reshape, Add, Concatenate
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.regularizers import l2
from layers.edm import EDM
from layers.comb_matrix import CombMatrix
from layers.normalization import InstanceNormalization

CONV2D_ARGS = {'padding': 'same', 'data_format': 'channels_last', 'kernel_regularizer': l2(5e-4)}


class _DMNN(object):
    def __init__(self, config):
        self.name = config.model_type + '_' + config.model_version
        self.data_set = config.data_set
        self.batch_size = config.batch_size
        self.num_actions = config.num_actions
        self.seq_len = config.pick_num if config.pick_num > 0 else (
                       config.crop_len if config.crop_len > 0 else None)
        self.njoints = config.njoints

        real_seq = Input(
            batch_shape=(self.batch_size, self.njoints, self.seq_len, 3),
            name='real_seq', dtype='float32')

        pred_action = self.classifier(real_seq)

        self.model = Model(real_seq, pred_action, name=self.name)
        self.model.compile(Adam(lr=config.learning_rate), 'sparse_categorical_crossentropy', ['accuracy'])


def _conv_block(x, out_filters, bneck_factor, kernel_size, strides, i, j, net_name, dilation_rate=(1, 1)):
    x = InstanceNormalization(axis=-1, name='%s/block_%d/branch_%d/inorm_in' % (net_name, i, j))(x)
    x = Activation('relu', name='%s/block_%d/branch_%d/relu_in' % (net_name, i, j))(x)
    x = Conv2D(filters=out_filters // bneck_factor, kernel_size=kernel_size, strides=1, dilation_rate=dilation_rate,
               name='%s/block_%d/branch_%d/conv_in' % (net_name, i, j), **CONV2D_ARGS)(x)
    x = InstanceNormalization(axis=-1, name='%s/block_%d/branch_%d/inorm_out' % (net_name, i, j))(x)
    x = Activation('relu', name='%s/block_%d/branch_%d/relu_out' % (net_name, i, j))(x)
    x = Conv2D(filters=out_filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,
               name='%s/block_%d/branch_%d/conv_out' % (net_name, i, j), **CONV2D_ARGS)(x)
    return x


class DMNNv1(_DMNN):
    # DM2DCNN

    def classifier(self, x):
        n_hidden = 128 if self.data_set == 'NTURGBD' else 64
        n_blocks = 3 if self.data_set == 'NTURGBD' else 2
        n_groups = 16 if self.data_set == 'NTURGBD' else 8
        strides = 2 if self.data_set == 'NTURGBD' else 3
        bneck_factor = 4 if self.data_set == 'NTURGBD' else 2

        x = CombMatrix(self.njoints, name='classifier/comb_matrix')(x)

        x = EDM(name='classifier/edms')(x)

        x = InstanceNormalization(axis=-1, name='classifier/inorm_in')(x)
        x = Activation('relu', name='classifier/relu_in')(x)

        x = Reshape((self.njoints * self.njoints, self.seq_len, 1), name='classifier/resh_in')(x)

        x = Conv2D(n_hidden // 2, 1, 1,
                   name='classifier/conv_in', **CONV2D_ARGS)(x)
        for i in range(n_blocks):
            n_filters = n_hidden * (2 ** i)
            shortcut = Conv2D(n_filters, strides, strides,
                        name='classifier/block_%d/shortcut' % i, **CONV2D_ARGS)(x)
            pis = []
            group_size = int(x.shape[-1]) // n_groups
            for j in range(n_groups):
                x_group = Lambda(lambda arg: arg[:, :, :, j * group_size: (j + 1) * group_size])(x)
                pis.append(_conv_block(x_group, n_filters // n_groups, bneck_factor, 3, strides, i, j,'classifier'))

            pi = Concatenate(name='classifier/block_%d/cat_pi' % i)(pis)

            x = Add(name='classifier/block_%d/add' % i)([shortcut, pi])

        x = Lambda(lambda args: K.mean(args, axis=(1, 2)), name='classifier/mean_pool')(x)
        x = Dense(self.num_actions, activation='softmax', name='classifier/label_out')(x)

        return x

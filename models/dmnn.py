from __future__ import absolute_import, division, print_function
import tensorflow.contrib.keras.api.keras.backend as K
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Input
from tensorflow.contrib.keras.api.keras.layers import Conv2D, \
    Dense, Activation, Lambda, Reshape, Add, Concatenate, \
    BatchNormalization, Dropout
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.regularizers import l2
from layers.edm import EDM
from layers.comb_matrix import CombMatrix

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


def _preact_conv(x, out_filters, kernel_size, strides, i, j, suffix):
    x = BatchNormalization(axis=-1, name='classifier/block_%d/branch_%d/bn_%s' % (i, j, suffix))(x)
    x = Activation('relu', name='classifier/block_%d/branch_%d/relu_%s' % (i, j, suffix))(x)
    x = Conv2D(filters=out_filters, kernel_size=kernel_size, strides=strides,
               name='classifier/block_%d/branch_%d/conv_%s' % (i, j, suffix), **CONV2D_ARGS)(x)
    return x


def _conv_block(x, out_filters, bneck_factor, n_groups, kernel_size, strides, i):
    shortcut = Conv2D(out_filters, strides, strides,
                      name='classifier/block_%d/shortcut' % i, **CONV2D_ARGS)(x)

    pi = _preact_conv(x, out_filters // bneck_factor, 1, 1, i, 0, 'in')

    pis = []
    group_size = int(pi.shape[-1]) // n_groups
    for j in range(n_groups):
        pi_group = Lambda(lambda arg: arg[:, :, :, j * group_size: (j + 1) * group_size],
                          name='classifier/block_%d/branch_%d/split_in' % (i, j))(pi)
        pis.append(_preact_conv(pi_group, (out_filters // bneck_factor) // n_groups,
                                kernel_size, strides, i, j, 'neck'))

    pi = Concatenate(name='classifier/block_%d/cat_pi' % i)(pis)

    pi = _preact_conv(pi, out_filters, 1, 1, i, 0, 'out')

    x = Add(name='classifier/block_%d/add' % i)([shortcut, pi])
    return x


class DMNNv1(_DMNN):
    # DM2DCNN (ResNext based)

    def classifier(self, x):
        n_hidden = 128
        n_blocks = 3
        n_groups = 16
        kernel_size = 3
        strides = 2
        bneck_factor = 4

        x = CombMatrix(self.njoints, name='classifier/comb_matrix')(x)

        x = EDM(name='classifier/edms')(x)

        x = Reshape((self.njoints * self.njoints, self.seq_len, 1), name='classifier/resh_in')(x)

        x = BatchNormalization(axis=-1, name='classifier/bn_in')(x)
        x = Conv2D(n_hidden // 2, 1, 1, name='classifier/conv_in', **CONV2D_ARGS)(x)
        for i in range(n_blocks):
            n_filters = n_hidden * (2 ** i)
            x = _conv_block(x, n_filters, bneck_factor, n_groups, kernel_size, strides, i)

        x = Lambda(lambda args: K.mean(args, axis=(1, 2)), name='classifier/mean_pool')(x)
        x = Dropout(0.5, name='classifier/dropout')(x)
        x = Dense(self.num_actions, activation='softmax', name='classifier/label_out')(x)

        return x

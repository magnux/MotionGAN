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
from utils.scoping import Scoping

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

    def update_lr(self, lr):
        K.set_value(self.model.optimizer.lr, lr)


def _preact_conv(x, out_filters, kernel_size, strides, groups=1):
    scope = Scoping.get_global_scope()
    x = BatchNormalization(axis=-1, name=scope+'bn')(x)
    x = Activation('relu', name=scope+'relu')(x)
    if groups > 1:
        branches = []
        group_size = int(x.shape[-1]) // groups
        for j in range(groups):
            with scope.name_scope('branch_%d' % j):
                x_group = Lambda(lambda arg: arg[:, :, :, j * group_size: (j + 1) * group_size], name=scope+'split')(x)
                branches.append(Conv2D(filters=out_filters // groups, kernel_size=kernel_size, strides=strides, name=scope+'conv', **CONV2D_ARGS)(x_group))
        x = Concatenate(name=scope+'cat')(branches)
    else:
        x = Conv2D(filters=out_filters, kernel_size=kernel_size, strides=strides, name=scope+'conv', **CONV2D_ARGS)(x)
    return x


def _conv_block(x, out_filters, bneck_filters, groups, kernel_size, strides):
    scope = Scoping.get_global_scope()
    shortcut = Conv2D(out_filters, strides, strides, name=scope+'shortcut', **CONV2D_ARGS)(x)

    with scope.name_scope('in'):
        pi = _preact_conv(x, bneck_filters, 1, 1)

    with scope.name_scope('bneck'):
        pi = _preact_conv(pi, bneck_filters, kernel_size, strides, groups)

    with scope.name_scope('out'):
        pi = _preact_conv(pi, out_filters, 1, 1)

    x = Add(name=scope+'add_shortcut')([shortcut, pi])
    return x


class DMNNv1(_DMNN):
    # DM2DCNN (ResNext based)

    def classifier(self, x):
        scope = Scoping.get_global_scope()
        with scope.name_scope('classifier'):
            if self.data_set == 'NTURGBD':
                blocks = [{'size': 128, 'bneck': 32, 'groups': 16, 'strides': 1},
                          {'size': 256, 'bneck': 64, 'groups': 16, 'strides': 2},
                          {'size': 512, 'bneck': 128, 'groups': 16, 'strides': 2}]
            else:
                blocks = [{'size': 64, 'bneck': 32, 'groups': 8, 'strides': 3},
                          {'size': 128, 'bneck': 64, 'groups': 8, 'strides': 3}]

            x = CombMatrix(self.njoints, name=scope+'comb_matrix')(x)

            x = EDM(name=scope+'edms')(x)

            x = Reshape((self.njoints * self.njoints, self.seq_len, 1), name=scope+'resh_in')(x)

            x = BatchNormalization(axis=-1, name=scope+'bn_in')(x)
            x = Conv2D(blocks[0]['bneck'], 1, 1, name=scope+'conv_in', **CONV2D_ARGS)(x)
            for i in range(len(blocks)):
                with scope.name_scope('block_%d' % i):
                    x = _conv_block(x, blocks[i]['size'], blocks[i]['bneck'],
                                    blocks[i]['groups'], 3, blocks[i]['strides'])

            x = Lambda(lambda args: K.mean(args, axis=(1, 2)), name=scope+'mean_pool')(x)
            x = Dropout(0.5, name=scope+'dropout')(x)
            x = Dense(self.num_actions, activation='softmax', name=scope+'label')(x)

        return x

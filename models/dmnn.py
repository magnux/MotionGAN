from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow.contrib.keras.api.keras.backend as K

from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Input
from tensorflow.contrib.keras.api.keras.layers import Conv2D, \
    Dense, Activation, Lambda, BatchNormalization, Reshape, Add
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.regularizers import l2

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


def _edm(x):
    x1 = K.expand_dims(x, axis=1)
    x2 = K.expand_dims(x, axis=2)
    # epsilon needed in sqrt to avoid numerical issues
    return K.sqrt(K.sum(K.square(x1 - x2), axis=-1) + K.epsilon())


class DMNNv1(_DMNN):
    # DM2DCNN

    def classifier(self, x):
        n_hidden = 32

        x = Lambda(lambda args: K.expand_dims(_edm(args), axis=-1), name='classifier/edms')(x)

        x = Reshape((self.njoints * self.njoints, self.seq_len, 1), name='classifier/resh_in')(x)

        x = Conv2D(n_hidden, 3, 1, name='classifier/conv_in', **CONV2D_ARGS)(x)
        for i in range(4):
            n_filters = n_hidden * (2 ** i)
            x = BatchNormalization(name='classifier/block_%d/bn_in' % i)(x)
            shortcut = Conv2D(n_filters, 2, 2,
                        name='classifier/block_%d/shortcut' % i, **CONV2D_ARGS)(x)
            pi = Conv2D(n_filters // 2, 3, 1, activation='relu',
                        name='classifier/block_%d/pi_0' % i, **CONV2D_ARGS)(x)
            pi = BatchNormalization(name='classifier/block_%d/pi_bn' % i)(pi)
            pi = Conv2D(n_filters, 3, 2, activation='relu',
                        name='classifier/block_%d/pi_1' % i, **CONV2D_ARGS)(pi)
            x = Add(name='classifier/block_%d/add' % i)([shortcut, pi])

        x = Lambda(lambda args: K.mean(args, axis=(1, 2)), name='classifier/mean_pool')(x)
        x = Dense(self.num_actions, activation='softmax')(x)

        return x

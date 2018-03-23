from __future__ import absolute_import, division, print_function
from tensorflow.contrib.keras.api.keras.layers import Layer
import tensorflow.contrib.keras.api.keras.backend as K


def edm(x, y=None):
    with K.name_scope('edm'):
        y = x if y is None else y
        x = K.expand_dims(x, axis=1)
        y = K.expand_dims(y, axis=2)
        return K.sqrt(K.sum(K.square(x - y), axis=-1) + K.epsilon())


def edm_loss(y_true, y_pred):
    return K.mean(K.sum(K.square(edm(y_true) - edm(y_pred)), axis=[1, 2]))


class EDM(Layer):
    def __init__(self, **kwargs):
        super(EDM, self).__init__(**kwargs)

    def build(self, input_shape):
        super(EDM, self).build(input_shape)

    def call(self, x):
        return edm(x)

    def compute_output_shape(self, input_shape):
        input_shape[3] = input_shape[2]
        input_shape[2] = input_shape[1]
        return input_shape


class Symmetry(Layer):
    def __init__(self, **kwargs):
        super(Symmetry, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Symmetry, self).build(input_shape)

    def call(self, x):
        return (x + K.permute_dimensions(x, (0, 2, 1, 3))) / 2

    def compute_output_shape(self, input_shape):
        return input_shape
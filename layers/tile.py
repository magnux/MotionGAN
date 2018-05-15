from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.layers import Layer


class Tile(Layer):
    def __init__(self, multiples, **kwargs):
        self.multiples = [int(i) for i in multiples]
        super(Tile, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Tile, self).build(input_shape)

    def call(self, x, **kwargs):
        x = tf.tile(x, [1] + self.multiples)
        return x

    def compute_output_shape(self, input_shape):
        int_shape = [int(i) for i in input_shape]
        return [int_shape[0]] + [i * j for i, j in zip(int_shape[1:], self.multiples)]

    def get_config(self):
        config = {
            'multiples': self.multiples,
        }
        return config

from __future__ import absolute_import, division, print_function

import numpy as np
from tensorflow.contrib.keras.api.keras.layers import Layer
from tensorflow.contrib.keras.api.keras.backend import conv1d, conv2d, bias_add, concatenate, constant
from tensorflow.contrib.keras.api.keras.utils import serialize_keras_object
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils

class _CausalConv(Layer):
    def __init__(self, rank, filters, kernel_size, strides, padding,
                 causal_dim, dilation_rate, data_format,
                 kernel_initializer, kernel_regularizer,
                 bias_initializer, bias_regularizer, **kwargs):
        self.rank = rank
        self.filters = filters
        self.kernel_size = (kernel_size,) * rank if isinstance(kernel_size, int) else kernel_size
        self.strides = (strides,) * rank if isinstance(strides, int) else strides
        self.padding = padding
        self.causal_dim = causal_dim if causal_dim >= 0 else rank - causal_dim
        self.dilation_rate = (dilation_rate,) * rank if isinstance(dilation_rate, int) else dilation_rate
        self.data_format = data_format
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        if rank == 1:
            self.conv = conv1d
        elif rank == 2:
            self.conv = conv2d
        else:
            raise Exception('unsupported rank for gated conv:', rank)
        super(_CausalConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = int(input_shape[-1])

        kernel_shape = [size for size in self.kernel_size]
        kernel_shape[self.causal_dim] = (self.kernel_size[self.causal_dim] // 2) + 1
        kernel_shape += [self.input_dim, self.filters]
        kernel = self.add_weight(name='kernel',
                                 shape=kernel_shape,
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 trainable=True)

        pad_shape = [size for size in self.kernel_size]
        pad_shape[self.causal_dim] = self.kernel_size[self.causal_dim] - kernel_shape[self.causal_dim]
        pad_shape += [self.input_dim, self.filters]
        kernel_pad = constant(np.zeros(pad_shape), dtype='float32')
        self.kernel = concatenate([kernel, kernel_pad], axis=self.causal_dim)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    trainable=True)
        super(_CausalConv, self).build(input_shape)

    def call(self, x, **kwargs):

        x = self.conv(x, self.kernel, strides=self.strides, padding=self.padding,
                      data_format=self.data_format, dilation_rate=self.dilation_rate)
        x = bias_add(x, self.bias)
        return x

    def _compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0]] + new_space + [self.filters])

    def get_config(self):
        config = {
            'rank': self.rank,
            'kernel_size': self.kernel_size,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
        }
        return config


class CausalConv1D(_CausalConv):
    def __init__(self, filters, kernel_size, strides, padding='same',
                 dilation_rate=1, data_format='channels_last',
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zeros', bias_regularizer=None, **kwargs):
        super(CausalConv1D, self).__init__(rank=1, filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           causal_dim=0,
                                           dilation_rate=dilation_rate,
                                           data_format=data_format,
                                           kernel_initializer=kernel_initializer,
                                           kernel_regularizer=kernel_regularizer,
                                           bias_initializer=bias_initializer,
                                           bias_regularizer=bias_regularizer,
                                           **kwargs)


class CausalConv2D(_CausalConv):
    def __init__(self, filters, kernel_size, strides, padding='same',
                 causal_dim=0, dilation_rate=1, data_format='channels_last',
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 bias_initializer='zeros', bias_regularizer=None, **kwargs):
        super(CausalConv2D, self).__init__(rank=2, filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           causal_dim=causal_dim,
                                           dilation_rate=dilation_rate,
                                           data_format=data_format,
                                           kernel_initializer=kernel_initializer,
                                           kernel_regularizer=kernel_regularizer,
                                           bias_initializer=bias_initializer,
                                           bias_regularizer=bias_regularizer,
                                           **kwargs)
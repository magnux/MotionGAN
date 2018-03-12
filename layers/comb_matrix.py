from __future__ import absolute_import, division, print_function
import numpy as np
from tensorflow.contrib.keras.api.keras.layers import Layer
from tensorflow.contrib.keras.api.keras.backend import dot, \
    permute_dimensions, reshape
from tensorflow.contrib.keras.api.keras.regularizers import l2
from tensorflow.contrib.keras.api.keras.initializers import Constant

class CombMatrix(Layer):
    def __init__(self, njoints, joints_dim=1, **kwargs):
        self.njoints = njoints
        self.joints_dim = joints_dim
        super(CombMatrix, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        init_mat = np.eye(self.njoints) + np.random.normal(0.0, 1e-2, (self.njoints, self.njoints))
        self.comb_matrix = self.add_weight(name='comb_matrix',
                                           shape=(self.njoints, self.njoints),
                                           initializer=Constant(init_mat),
                                           regularizer=l2(5e-4),
                                           trainable=True)
        super(CombMatrix, self).build(input_shape)

    def call(self, x):
        perm_dims = range(len(self.shape))
        perm_dims[self.joints_dim], perm_dims[-1] = perm_dims[-1], perm_dims[self.joints_dim]
        x = permute_dimensions(x, perm_dims)
        # x = reshape(x, [np.prod([int(self.shape[i])
        #                          for i in range(len(self.shape))
        #                          if i != self.joints_dim]),
        #                 int(self.shape[self.joints_dim])])
        x = dot(x, self.comb_matrix)
        # x = reshape(x, [int(self.shape[i])
        #                 for i in range(len(self.shape))
        #                 if i != self.joints_dim] +
        #                 [int(self.shape[self.joints_dim])])
        x = permute_dimensions(x, perm_dims)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'njoints': self.njoints,
            'joints_dim': self.joints_dim,
        }
        return config
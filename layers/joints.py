from __future__ import absolute_import, division, print_function
import numpy as np
from tensorflow.contrib.keras.api.keras.layers import Layer
from tensorflow.contrib.keras.api.keras.backend import dot, constant, permute_dimensions

# NTURGBD
joint_order_ntu = np.array([21, 9, 10, 11, 12, 24, 25, 24, 12, 11, 10, 9,  # Left Arm
                            21, 3, 4, 3,  # Head
                            21, 5, 6, 7, 8, 22, 23, 22, 8, 7, 6, 5,  # Right Arm
                            21, 2, 1,  # Torso
                            17, 18, 19, 20, 19, 18, 17, 1,  # Left Leg
                            13, 14, 15, 16, 15, 14, 13, 1])  # Right Leg

joint_order_ntu -= 1  # Reindexing joints

# Human3.6 (Original version, with static joints)
# joint_order_h36 = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1,  # Left Leg
#                         0, 6, 7, 8, 9, 10, 9, 8, 7, 6,  # Right Leg
#                         0, 11, 12,  # Torso
#                         13, 14, 15, 14, 13, 12,  # Head
#                         16, 17, 18, 19, 20, 21, 20, 19, 22, 23, 22, 19, 18, 17, 16, 12,  # Left Arm and Hand
#                         24, 25, 26, 27, 28, 29, 28, 27, 30, 31, 30, 27, 26, 25, 24, 12])  # Right Arm and Hand

# Human3.6 (3D pose baseline paper version)
joint_order_h36 = np.array([6, 0, 1, 2, 1, 0,  # Left Leg
                            6, 3, 4, 5, 4, 3,  # Right Leg
                            6, 7,  # Torso
                            8, 9, 8, 7,  # Head
                            10, 11, 12, 11, 10, 7,  # Left Arm and Hand
                            13, 14, 15, 14, 13, 7])  # Right Arm and Hand

class UnfoldJoints(Layer):
    def __init__(self, data_set, **kwargs):
        if data_set == 'NTURGBD':
            self.joint_order = joint_order_ntu
        elif data_set == 'HUMAN36':
            self.joint_order = joint_order_h36

        self.njoints = np.max(self.joint_order) + 1

        super(UnfoldJoints, self).__init__(**kwargs)

    def build(self, input_shape):
        convert_mat = np.zeros([self.njoints, len(self.joint_order)], dtype=np.float32)
        for i, j in enumerate(self.joint_order):
            convert_mat[j, i] = 1.0

        self.convert_mat = constant(convert_mat)
        super(UnfoldJoints, self).build(input_shape)

    def call(self, x):
        x = permute_dimensions(x, (0, 3, 2, 1))
        x = dot(x, self.convert_mat)
        return permute_dimensions(x, (0, 3, 2, 1))

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape[-1] = self.convert_mat.shape[-1]
        return output_shape

    def get_config(self):
        config = {
            'njoints': self.njoints
        }
        return config


class FoldJoints(Layer):
    def __init__(self, data_set, **kwargs):
        if data_set == 'NTURGBD':
            self.joint_order = joint_order_ntu
        elif data_set == 'HUMAN36':
            self.joint_order = joint_order_h36

        self.njoints = np.max(self.joint_order) + 1

        super(FoldJoints, self).__init__(**kwargs)

    def build(self, input_shape):
        convert_mat = np.zeros([len(self.joint_order), self.njoints], dtype=np.float32)
        for i, j in enumerate(self.joint_order):
            convert_mat[i, j] = 1.0

        self.norm_fact = constant(np.sum(convert_mat, 0))
        self.convert_mat = constant(convert_mat)
        super(FoldJoints, self).build(input_shape)

    def call(self, x):
        x = permute_dimensions(x, (0, 3, 2, 1))
        x = dot(x, self.convert_mat) / self.norm_fact
        return permute_dimensions(x, (0, 3, 2, 1))

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape[-1] = self.convert_mat.shape[-1]
        return output_shape

    def get_config(self):
        config = {
            'njoints': self.njoints
        }
        return config
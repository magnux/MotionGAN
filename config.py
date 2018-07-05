from __future__ import absolute_import, division, print_function

import os
import pickle
import tensorflow as tf
import numpy as np
from collections import OrderedDict


class Config(object):
    def __init__(self, flags):
        self.base_config = os.path.join('configs', 'base_config.py')
        with open(self.base_config, 'r') as f:
            dict_file = eval(f.read())
            self.__dict__ = dict_file

        self.save_path = flags.save_path
        self.saved_config = (self.save_path + '_config.pickle') if self.save_path is not None else None
        self.template_config = os.path.join('configs', flags.config_file + '_config.py') if flags.config_file is not None else None

        if self.saved_config is not None:
            assert tf.gfile.Exists(self.saved_config),\
                '%s config file not found!' % self.saved_config
            with open(self.saved_config, 'rb') as f:
                pickle_file = pickle.load(f)
                del pickle_file.__dict__['save_path']  # This allows to modify the path after training
                del pickle_file.__dict__['saved_config']
                self.__dict__.update(pickle_file.__dict__)
        elif self.template_config is not None:
            assert tf.gfile.Exists(self.template_config),\
                '%s config file not found!' % self.template_config
            with open(self.template_config, 'r') as f:
                dict_file = eval(f.read())
                self.__dict__.update(dict_file)

        self.save_path = self.save_path if self.save_path is not None else 'save/%s' % (flags.config_file)
        self.saved_config = self.saved_config if self.saved_config is not None else (self.save_path + '_config.pickle')

    def save(self):
        with open(self.saved_config, 'wb') as f:
            pickle.dump(self, f)


def get_config(flags):
    assert flags.config_file is not None or flags.save_path is not None,\
        'Either config_file or save_path must be provided'
    config = Config(flags)

    config.epoch = config.epoch if hasattr(config, 'epoch') else 0
    config.batch = config.batch if hasattr(config, 'batch') else 0
    config.nan_restarts = config.nan_restarts if hasattr(config, 'nan_restarts') else 0
    config.only_val = config.only_val if hasattr(config, 'only_val') else False
    config.best_err = np.inf
    config.best_epoch = 0

    if config.data_set == 'NTURGBD':
        config.num_actions = 60
        config.num_subjects = 40
        config.njoints = 25  # *2, Note: only taking first skeleton
        config.max_plen = 300
        config.body_members = {
            'left_arm': {'joints': [20, 8, 9, 10, 11, 23], 'side': 'left'},
            'left_fingers': {'joints': [11, 24], 'side': 'left'},
            'right_arm': {'joints': [20, 4, 5, 6, 7, 21], 'side': 'right'},
            'right_fingers': {'joints': [7, 22], 'side': 'right'},
            'head': {'joints': [20, 2, 3], 'side': 'right'},
            'torso': {'joints': [0, 1, 20], 'side': 'right'},
            'left_leg': {'joints': [0, 16, 17, 18, 19], 'side': 'left'},
            'right_leg': {'joints': [0, 12, 13, 14, 15], 'side': 'right'},
        }
    elif config.data_set == 'MSRC12':
        config.num_actions = 12
        config.num_subjects = 30
        config.njoints = 20
        config.max_plen = 1320
        config.body_members = {
            'left_arm': {'joints': [2, 4, 5, 6, 7], 'side': 'left'},
            'right_arm': {'joints': [2, 8, 9, 10, 11], 'side': 'right'},
            'head': {'joints': [1, 2, 3], 'side': 'right'},
            'torso': {'joints': [0, 1], 'side': 'right'},
            'left_leg': {'joints': [0, 12, 13, 14, 15], 'side': 'left'},
            'right_leg': {'joints': [0, 16, 17, 18, 19], 'side': 'right'},
        }
    elif config.data_set == 'Human36':
        config.num_actions = 15
        config.num_subjects = 7
        # config.max_plen = (6343 // 2) + 1  # Data will be subsampled to 25hz
        config.max_plen = (6343 // 10) + 1  # Data will be subsampled to 5hz

        config.body_members = {
            'left_arm': {'joints': [13, 17, 18, 19], 'side': 'left'},
            'right_arm': {'joints': [13, 25, 26, 27], 'side': 'right'},
            'head': {'joints': [13, 14, 15], 'side': 'right'},
            'torso': {'joints': [0, 12, 13], 'side': 'right'},
            'left_leg': {'joints': [0, 6, 7, 8], 'side': 'left'},
            'right_leg': {'joints': [0, 1, 2, 3], 'side': 'right'},
        }
        config.used_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
        config.njoints = len(config.used_joints)
        config.full_njoints = 32
        new_body_members = {}
        for key, value in config.body_members.items():
            new_body_members[key] = value.copy()
            new_body_members[key]['joints'] = [config.used_joints.index(j) for j in new_body_members[key]['joints']]
        config.body_members = new_body_members

        config.full_body_members = {
            'left_arm': {'joints': [13, 16, 17, 18, 19, 20, 21], 'side': 'left'},
            'left_fingers': {'joints': [19, 22, 23], 'side': 'left'},
            'right_arm': {'joints': [13, 24, 25, 26, 27, 28, 29], 'side': 'right'},
            'right_fingers': {'joints': [27, 30, 31], 'side': 'right'},
            'head': {'joints': [13, 14, 15], 'side': 'right'},
            'torso': {'joints': [0, 11, 12, 13], 'side': 'right'},
            'left_leg': {'joints': [0, 6, 7, 8, 9, 10], 'side': 'left'},
            'right_leg': {'joints': [0, 1, 2, 3, 4, 5], 'side': 'right'},
        }

    elif config.data_set == 'Human36_expmaps':
        config.num_actions = 15
        config.num_subjects = 7
        # config.max_plen = (6343 // 2) + 1  # Data will be subsampled to 25hz
        config.max_plen = (6343 // 10) + 1  # Data will be subsampled to 5hz

        config.body_members = {
            'left_arm': {'joints': [13, 16, 17, 18, 19, 20, 21], 'side': 'left'},
            'left_fingers': {'joints': [19, 22, 23], 'side': 'left'},
            'right_arm': {'joints': [13, 24, 25, 26, 27, 28, 29], 'side': 'right'},
            'right_fingers': {'joints': [27, 30, 31], 'side': 'right'},
            'head': {'joints': [13, 14, 15], 'side': 'right'},
            'torso': {'joints': [0, 11, 12, 13], 'side': 'right'},
            'left_leg': {'joints': [0, 6, 7, 8, 9, 10], 'side': 'left'},
            'right_leg': {'joints': [0, 1, 2, 3, 4, 5], 'side': 'right'},
        }
        config.used_joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 25, 26, 27, 28]
        config.njoints = len(config.used_joints)
        config.full_njoints = 33
        new_body_members = {}
        for key, value in config.body_members.items():
            new_body_members[key] = value.copy()
            new_body_members[key]['joints'] = [config.used_joints.index(j) for j in new_body_members[key]['joints'] if j in config.used_joints]
        config.body_members = new_body_members

    config.body_members = OrderedDict(sorted(config.body_members.iteritems()))  # Ordering might be important for iter

    return config

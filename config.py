from __future__ import absolute_import, division, print_function

import os
import pickle
import tensorflow as tf


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
    config.only_val = config.only_val if hasattr(config, 'only_val') else False

    if config.data_set == 'NTURGBD':
        config.num_actions = 60
        config.num_subjects = 40
        config.njoints = 25  # *2, Note: only taking first skeleton
        config.max_plen = 300
        config.body_members = {
            'left_arm': {'joints': [20, 8, 9, 10, 11], 'side': 'left'},
            # [21, 9, 10, 11, 12, 24, 25]
            'right_arm': {'joints': [20, 4, 5, 6, 7], 'side': 'right'},
            # [21, 5, 6, 7, 8, 22, 23]
            'head': {'joints': [20, 2, 3], 'side': 'right'},
            'torso': {'joints': [20, 1, 0], 'side': 'right'},
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
            'torso': {'joints': [1, 0], 'side': 'right'},
            'left_leg': {'joints': [0, 12, 13, 14, 15], 'side': 'left'},
            'right_leg': {'joints': [0, 16, 17, 18, 19], 'side': 'right'},
        }

    return config

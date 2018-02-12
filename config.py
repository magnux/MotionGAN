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
    config.only_val = config.only_val if hasattr(config, 'only_val') else False
    config.no_val = config.no_val if hasattr(config, 'no_val') else True

    if config.data_set == 'NTURGBD':
        config.num_actions = 60
        config.num_subjects = 40
        config.njoints = 25  # *2, Note: only taking first skeleton
        config.max_plen = 300

    return config

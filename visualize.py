from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from config import get_config
from data_input import DataInput
from utils.viz import plot_gif

logging = tf.logging
flags = tf.flags
flags.DEFINE_bool("verbose", False, "To talk or not to talk")
flags.DEFINE_string("save_path", None, "Model output directory")
flags.DEFINE_string("config_file", "motiongan_v1", "Model config file")
FLAGS = flags.FLAGS

if __name__ == "__main__":
    # Config stuff
    config = get_config(FLAGS)
    config.only_val = True
    # config.pick_num = 0
    data_input = DataInput(config)

    labs_batch, poses_batch = data_input.batch_generator(False).next()

    print(np.shape(poses_batch), np.shape(labs_batch))

    rand_indices = np.random.permutation(config.batch_size)

    for j in range(config.batch_size):

        seq_idx = rand_indices[j]

        plot_gif(poses_batch[seq_idx, ...], poses_batch[seq_idx, ...], labs_batch[seq_idx, ...], config.data_set)

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from config import get_config
from data_input import DataInput
from utils.viz import plot_gif, plot_mult_gif

logging = tf.logging
flags = tf.flags
flags.DEFINE_bool("verbose", False, "To talk or not to talk")
flags.DEFINE_string("save_path", None, "Model output directory")
flags.DEFINE_string("config_file", "motiongan_v1", "Model config file")
FLAGS = flags.FLAGS

if __name__ == "__main__":
    # Config stuff
    config = get_config(FLAGS)
    # config.only_val = True
    config.normalize_data = False
    # config.pick_num = 0
    data_input = DataInput(config)

    n_batches = 4
    n_splits = 8
    print('Plotting %d batches in %d splits for the %s dataset' %
          (n_batches, n_splits, config.data_set))
    for b in range(n_batches):

        labs_batch, poses_batch = data_input.batch_generator(False).next()

        n_seqs = (config.batch_size // n_splits)
        for i in range(n_splits):
            plot_mult_gif(poses_batch[i * n_seqs:(i + 1) * n_seqs, ...],
                          labs_batch[i * n_seqs:(i + 1) * n_seqs, ...],
                          config.data_set, 'save/%s_viz_%d%d.gif' % (config.data_set, b, i))
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from config import get_config
from data_input import DataInput
from utils.viz import plot_seq_gif
from tqdm import trange

logging = tf.logging
flags = tf.flags
flags.DEFINE_bool("verbose", False, "To talk or not to talk")
flags.DEFINE_string("save_path", None, "Model output directory")
flags.DEFINE_string("config_file", "motiongan_v1_fae_h36", "Model config file")
FLAGS = flags.FLAGS


def _reset_rand_seed():
    seed = 42
    np.random.seed(seed)


if __name__ == "__main__":
    # Config stuff
    config = get_config(FLAGS)
    # config.only_val = True
    config.normalize_data = False
    # config.pick_num = 0
    data_input = DataInput(config)
    _reset_rand_seed()

    n_batches = 4
    n_splits = 32
    print('Plotting %d batches in %d splits for the %s dataset' %
          (n_batches, n_splits, config.data_set))
    for b in range(n_batches):

        labs_batch, poses_batch = data_input.batch_generator(False).next()

        n_seqs = (config.batch_size // n_splits)
        for i in trange(n_splits):
            plot_seq_gif(poses_batch[i * n_seqs:(i + 1) * n_seqs, :, :, :3],
                         labs_batch[i * n_seqs:(i + 1) * n_seqs, ...],
                         config.data_set,
                         # save_path='save/vis_%s_%d_%d.gif' % (config.data_set, b, i),
                         figwidth=1920, figheight=1080)
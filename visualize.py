from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from config import get_config
from data_input import DataInput
import random

logging = tf.logging
flags = tf.flags
flags.DEFINE_bool("verbose", False, "To talk or not to talk")
flags.DEFINE_string("save_path", None, "Model output directory")
flags.DEFINE_string("config_file", "motiongan_v1", "Model config file")
FLAGS = flags.FLAGS

NTU_ACTIONS = ["drink water", "eat meal/snack", "brushing teeth",
               "brushing hair", "drop", "pickup", "throw", "sitting down",
               "standing up (from sitting position)", "clapping", "reading",
               "writing", "tear up paper", "wear jacket", "take off jacket",
               "wear a shoe", "take off a shoe", "wear on glasses",
               "take off glasses", "put on a hat/cap", "take off a hat/cap",
               "cheer up", "hand waving", "kicking something",
               "put something inside pocket / take out something from pocket",
               "hopping (one foot jumping)", "jump up",
               "make a phone call/answer phone", "playing with phone/tablet",
               "typing on a keyboard", "pointing to something with finger",
               "taking a selfie", "check time (from watch)",
               "rub two hands together", "nod head/bow", "shake head",
               "wipe face", "salute", "put the palms together",
               "cross hands in front (say stop)", "sneeze/cough", "staggering",
               "falling", "touch head (headache)",
               "touch chest (stomachache/heart pain)", "touch back (backache)",
               "touch neck (neckache)", "nausea or vomiting condition",
               "use a fan (with hand or paper)/feeling warm",
               "punching/slapping other person", "kicking other person",
               "pushing other person", "pat on back of other person",
               "point finger at the other person", "hugging other person",
               "giving something to other person",
               "touch other person's pocket", "handshaking",
               "walking towards each other", "walking apart from each other"]

if __name__ == "__main__":
    # Config stuff
    config = get_config(FLAGS)
    config.only_val = True
    # config.pick_num = 0
    data_input = DataInput(config)

    labs_batch, poses_batch = data_input.batch_generator(False).next()

    print(np.shape(poses_batch), np.shape(labs_batch))

    rand_indices = np.random.permutation(config.batch_size)

    import matplotlib.pyplot as plt
    import utils.viz as viz

    # === Plot and animate ===
    fig = plt.figure(dpi=160)
    ax = plt.gca(projection='3d')
    ax.view_init(elev=90, azim=-90)
    # ax.view_init(elev=0, azim=90)
    # plt.subplot(1, 3, 1)
    ob = viz.Ax3DPose(plt.gca())

    njoints = 25
    ncams = 3

    for j in range(config.batch_size):
        seq_idx = rand_indices[j]
        poses = poses_batch[seq_idx, ...]
        seq_idx, subject, action, plen = labs_batch[seq_idx, ...]

        print("action: %s  subject: %d  seq_idx: %d  length: %d" %
              (NTU_ACTIONS[action], subject, seq_idx, plen))
        for i in range(plen):
            ob.update(poses[:, :, i])

            plt.show(block=False)
            fig.canvas.draw()

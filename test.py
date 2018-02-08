from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from config import get_config
from data_input import DataInput
from models.motiongan import MotionGANV1, MotionGANV2
from utils.restore_keras_model import restore_keras_model


logging = tf.logging
flags = tf.flags
flags.DEFINE_bool("verbose", False, "To talk or not to talk")
flags.DEFINE_string("save_path", None, "Model output directory")
flags.DEFINE_string("config_file", None, "Model config file")
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
    val_batches = data_input.train_epoch_size
    val_generator = data_input.batch_generator(False)

    # Model building
    if config.model_type == 'motiongan':
        if config.model_version == 'v1':
            model_wrap = MotionGANV1(config)
        if config.model_version == 'v2':
            model_wrap = MotionGANV2(config)

    if FLAGS.verbose:
        print('Discriminator model:')
        print(model_wrap.disc_model.summary())
        print('Generator model:')
        print(model_wrap.gen_model.summary())
        print('GAN model:')
        print(model_wrap.gan_model.summary())

    assert config.epoch > 0, 'Nothing to test in an untrained model'

    model_wrap.disc_model = restore_keras_model(
        model_wrap.disc_model, config.save_path + '_disc_weights.hdf5', False)
    model_wrap.gen_model = restore_keras_model(
        model_wrap.gen_model, config.save_path + '_gen_weights.hdf5', False)

    labs_batch, poses_batch = val_generator.next()

    gen_inputs = [poses_batch]
    if config.latent_cond_dim > 0:
        latent_noise = np.random.uniform(
            size=(config.batch_size, config.latent_cond_dim))
        gen_inputs.append(latent_noise)
    gen_outputs = model_wrap.gen_model.predict(gen_inputs, config.batch_size)

    poses_batch = gen_outputs

    import matplotlib.pyplot as plt
    import utils.viz as viz

    # === Plot and animate ===
    fig = plt.figure(dpi=160)
    ax = plt.gca(projection='3d')
    ax.view_init(elev=90, azim=-90)
    # ax.view_init(elev=0, azim=90)
    # plt.subplot(1, 3, 1)
    ob = viz.Ax3DPose(plt.gca())

    rand_indices = np.random.permutation(config.batch_size)

    for j in range(config.batch_size):
        seq_idx = rand_indices[j]
        poses = poses_batch[seq_idx, ...]
        seq_idx, subject, action, plen = labs_batch[seq_idx, ...]

        print("action: %s  subject: %d  seq_idx: %d  length: %d" %
              (NTU_ACTIONS[action], subject, seq_idx, plen))
        for i in range(plen):
            ob.update(poses[:, i, :])

            plt.show(block=False)
            fig.canvas.draw()

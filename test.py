from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from config import get_config
from data_input import DataInput
from models.motiongan import MotionGANV1, MotionGANV2
from utils.restore_keras_model import restore_keras_model
from utils.viz import plot_gif
from scipy.linalg import pinv
from scipy.fftpack import idct

logging = tf.logging
flags = tf.flags
flags.DEFINE_bool("verbose", False, "To talk or not to talk")
flags.DEFINE_string("save_path", None, "Model output directory")
flags.DEFINE_string("config_file", None, "Model config file")
FLAGS = flags.FLAGS

if __name__ == "__main__":
    # Config stuff
    config = get_config(FLAGS)
    config.only_val = True
    config.batch_size = 4
    # config.pick_num = 0
    data_input = DataInput(config)
    val_batches = data_input.val_epoch_size
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

    mask_modes = ['No mask', 'Future Prediction', 'Oclusion Simulation', 'Noisy Transmission']

    def gen_mask(mask_type=0, keep_prob=1.0):
        # Default mask, no mask
        mask = np.ones(shape=(config.batch_size, config.njoints, model_wrap.seq_len, 1))
        if mask_type == 1:  # Future Prediction
            mask[:, :, np.int(model_wrap.seq_len * keep_prob):, :] = 0.0
        elif mask_type == 2:  # Occlusion Simulation
            rand_joints = np.random.randint(config.njoints, size=np.int(config.njoints * (1.0 - keep_prob)))
            mask[:, rand_joints, :, :] = 0.0
        elif mask_type == 3:  # Noisy transmission
            mask = np.random.binomial(1, keep_prob, size=mask.shape)

        return mask

    # Q = idct(np.eye(10))[:3, :]
    # Q_inv = pinv(Q)
    # Qs = np.matmul(Q_inv, Q)

    while True:
        labs_batch, poses_batch = val_generator.next()

        mask_mode = np.random.randint(4)
        mask_batch = gen_mask(mask_mode, 0.5)
        gen_inputs = [poses_batch, mask_batch]

        if config.latent_cond_dim > 0:
            latent_noise = np.random.uniform(
                size=(config.batch_size, config.latent_cond_dim))
            gen_inputs.append(latent_noise)

        gen_outputs = model_wrap.gen_model.predict(gen_inputs, config.batch_size)

        poses_batch = data_input.denormalize_poses(poses_batch)
        gen_outputs = data_input.denormalize_poses(gen_outputs)

        rand_indices = np.random.permutation(config.batch_size)

        for j in range(config.batch_size):
            seq_idx = rand_indices[j]

            plot_gif(poses_batch[seq_idx, ...],
                     gen_outputs[seq_idx, ...],
                     labs_batch[seq_idx, ...],
                     config.data_set,
                     extra_text='mask mode: %s' % mask_modes[mask_mode],
                     seq_mask=mask_batch[seq_idx, ...])

            # Smoothing code
            # smoothed = np.transpose(gen_outputs[seq_idx, ...], (0, 2, 1))
            # smoothed = np.reshape(smoothed, (25 * 3, 20))
            # smoothed[:, 10:] = np.matmul(smoothed[:, 10:], Qs)
            # smoothed = np.reshape(smoothed, (25, 3, 20))
            # smoothed = np.transpose(smoothed, (0, 2, 1))
            # plot_gif(gen_outputs[seq_idx, ...], smoothed, labs_batch[seq_idx, ...])
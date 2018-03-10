from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from config import get_config
from data_input import DataInput
from models.motiongan import MotionGANV1, MotionGANV2, MotionGANV3, MotionGANV4
from utils.restore_keras_model import restore_keras_model
from utils.viz import plot_seq_gif
import h5py as h5
from tqdm import trange

MASK_MODES = ('No mask', 'Future Prediction', 'Oclusion Simulation', 'Noisy Transmission')

logging = tf.logging
flags = tf.flags
flags.DEFINE_bool("verbose", False, "To talk or not to talk")
flags.DEFINE_string("save_path", None, "Model output directory")
flags.DEFINE_string("config_file", None, "Model config file")
flags.DEFINE_string("test_mode", "show_images", "Test modes: show_images, write_images, write_data")
flags.DEFINE_integer("mask_mode", 0, "Mask modes: 0:%s, 1:%s, 2:%s, 3:%s" % MASK_MODES)
flags.DEFINE_float("keep_prob", 0.5, "Probability of keeping input data. (1 == Keep All)")
FLAGS = flags.FLAGS

if __name__ == "__main__":
    # Config stuff
    config = get_config(FLAGS)
    config.only_val = True
    if FLAGS.test_mode == "show_imgs":
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
        if config.model_version == 'v3':
            model_wrap = MotionGANV3(config)
        if config.model_version == 'v4':
            model_wrap = MotionGANV4(config)

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

    def get_inputs():
        labs_batch, poses_batch = val_generator.next()

        mask_batch = gen_mask(FLAGS.mask_mode, FLAGS.keep_prob)
        gen_inputs = [poses_batch, mask_batch]

        if config.latent_cond_dim > 0:
            latent_noise = np.random.uniform(
                size=(config.batch_size, config.latent_cond_dim))
            gen_inputs.append(latent_noise)

        return labs_batch, poses_batch, mask_batch, gen_inputs

    def apply_constant_motion(seq, mask):
        new_seq = seq * mask
        for j in range(seq.shape[0]):
            for f in range(seq.shape[1]):
                if mask[j, f, 0] == 0:
                    new_seq[j, f, :] = 2 * new_seq[j, f - 1, :] - new_seq[j, f - 2, :]
        return new_seq

    def apply_kalman_filter(seq, mask):
        new_seq = np.zeros_like(seq)
        return new_seq

    if "images" in FLAGS.test_mode:

        for i in trange(val_batches):
            labs_batch, poses_batch, mask_batch, gen_inputs = get_inputs()

            gen_outputs = model_wrap.gen_model.predict(gen_inputs, config.batch_size)

            if config.normalize_data:
                poses_batch = data_input.denormalize_poses(poses_batch)
                gen_outputs = data_input.denormalize_poses(gen_outputs)

            rand_indices = np.random.permutation(config.batch_size)

            for j in range(config.batch_size):
                seq_idx = rand_indices[j]

                save_path = None
                if FLAGS.test_mode == "write_images":
                    save_path = "%s_test_gifs/%d_%d.gif" % (config.save_path, i, j)
                plot_seq_gif(np.concatenate([poses_batch[np.newaxis, seq_idx, ...],
                                             gen_outputs[np.newaxis, seq_idx, ...]]),
                             labs_batch[seq_idx, ...],
                             config.data_set,
                             seq_masks=mask_batch[seq_idx, ...],
                             extra_text='mask mode: %s' % MASK_MODES[FLAGS.mask_mode],
                             save_path=save_path)

    elif FLAGS.test_mode == "write_data":
        data_split = 'Validate'

        h5file = h5.File("%s_%s_%s_%s_%d_%.1f.h5" %
                         (config.data_set, config.data_set_version,
                          config.model_type, config.model_version,
                          FLAGS.mask_mode, FLAGS.keep_prob), "w")

        for _ in trange(val_batches):

            labs_batch, poses_batch, mask_batch, gen_inputs = get_inputs()

            gen_outputs = model_wrap.gen_model.predict(gen_inputs, config.batch_size)

            if config.normalize_data:
                poses_batch = data_input.denormalize_poses(poses_batch)
                gen_outputs = data_input.denormalize_poses(gen_outputs)

            for j in range(config.batch_size):
                seq_idx, subject, action, plen = labs_batch[j, ...]

                sub_array = np.array(subject)
                act_array = np.array(action)
                pose_array = gen_outputs[j, ...]

                data_path = '%s/%s/SEQ%d/' % (config.data_set, data_split, seq_idx)
                h5file.create_dataset(
                    data_path + 'Subject', np.shape(sub_array),
                    dtype='int32', data=sub_array
                )
                h5file.create_dataset(
                    data_path + 'Action', np.shape(act_array),
                    dtype='int32', data=act_array
                )
                h5file.create_dataset(
                    data_path + 'Pose', np.shape(pose_array),
                    dtype='float32', data=pose_array
                )

        h5file.flush()
        h5file.close()

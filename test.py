from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from config import get_config
from data_input import DataInput
from models.motiongan import MotionGANV1, MotionGANV2, MotionGANV3, MotionGANV4
from models.dmnn import DMNNv1
from utils.restore_keras_model import restore_keras_model
from utils.viz import plot_seq_gif, plot_seq_pano
import h5py as h5
from tqdm import trange

MASK_MODES = ('No mask', 'Future Prediction', 'Occlusion Simulation', 'Structured Occlusion', 'Noisy Transmission')

logging = tf.logging
flags = tf.flags
flags.DEFINE_bool("verbose", False, "To talk or not to talk")
flags.DEFINE_string("save_path", None, "Model output directory")
flags.DEFINE_string("config_file", None, "Model config file")
flags.DEFINE_string("test_mode", "show_images", "Test modes: show_images, write_images, write_data, dmnn_score")
flags.DEFINE_string("dmnn_save_path", None, "Path to trained DMNN model")
flags.DEFINE_string("images_mode", "gif", "Image modes: gif, png")
flags.DEFINE_integer("mask_mode", 3, "Mask modes: 0:%s, 1:%s, 2:%s, 3:%s, 4:%s" % MASK_MODES)
flags.DEFINE_float("keep_prob", 0.8, "Probability of keeping input data. (1 == Keep All)")
FLAGS = flags.FLAGS

if __name__ == "__main__":
    # Config stuff
    config = get_config(FLAGS)
    config.only_val = True
    if "images" in FLAGS.test_mode:
        config.batch_size = 16
    if FLAGS.test_mode == "write_images":
        images_path = "%s_test_images_%s/" % (config.save_path, FLAGS.images_mode)
        if not tf.gfile.Exists(images_path):
            tf.gfile.MkDir(images_path)
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
        elif mask_type == 3:  # Structured Occlusion Simulation
            rand_joints = set()
            while ((config.njoints - len(rand_joints)) >
                   (config.njoints * keep_prob)):
                joints_to_add = (config.body_members.values()[np.random.randint(len(config.body_members))])['joints']
                for joint in joints_to_add:
                    rand_joints.add(joint)
            print(list(rand_joints))
            mask[:, list(rand_joints), :, :] = 0.0
        elif mask_type == 4:  # Noisy transmission
            mask = np.random.binomial(1, keep_prob, size=mask.shape)

        return mask

    def get_inputs():
        labs_batch, poses_batch = val_generator.next()

        mask_batch = poses_batch[..., 3, np.newaxis]
        mask_batch = mask_batch * gen_mask(FLAGS.mask_mode, FLAGS.keep_prob)
        poses_batch = poses_batch[..., :3]

        gen_inputs = [poses_batch, mask_batch]

        if config.latent_cond_dim > 0:
            latent_noise = np.random.uniform(
                size=(config.batch_size, config.latent_cond_dim))
            gen_inputs.append(latent_noise)

        return labs_batch, poses_batch, mask_batch, gen_inputs

    def constant_baseline(seq, mask):
        new_seq = seq * mask
        new_seq[:, 0, :] = seq[:, 0, :]
        for j in range(seq.shape[0]):
            for f in range(1, seq.shape[1]):
                if mask[j, f, 0] == 0:
                    new_seq[j, f, :] = new_seq[j, f - 1, :]
        return new_seq

    def kalman_filter(seq, mask):
        new_seq = np.zeros_like(seq)
        return new_seq

    if "images" in FLAGS.test_mode:

        for i in trange(val_batches):
            labs_batch, poses_batch, mask_batch, gen_inputs = get_inputs()

            gen_outputs = model_wrap.gen_model.predict(gen_inputs, config.batch_size)

            if config.normalize_data:
                poses_batch = data_input.denormalize_poses(poses_batch)
                gen_outputs = data_input.denormalize_poses(gen_outputs)

            # rand_indices = np.random.permutation(config.batch_size)

            for j in range(config.batch_size):
                # seq_idx = rand_indices[j]
                seq_idx = j

                save_path = None
                if FLAGS.test_mode == "write_images":
                    save_path = images_path + ("%d_%d.%s" % (i, j, FLAGS.images_mode))

                if FLAGS.images_mode == "gif":
                    plot_func = plot_seq_gif
                    figwidth = 384 * 3
                    figheight = 384
                elif FLAGS.images_mode == "png":
                    plot_func = plot_seq_pano
                    figwidth = 768
                    figheight = 384 * 3

                constant_seq =\
                    constant_baseline(poses_batch[seq_idx, ...], mask_batch[seq_idx, ...])
                constant_seq = np.expand_dims(constant_seq, 0)

                plot_func(np.concatenate([poses_batch[np.newaxis, seq_idx, ...],
                                          gen_outputs[np.newaxis, seq_idx, ...],
                                          constant_seq]),
                          labs_batch[seq_idx, ...],
                          config.data_set,
                          seq_masks=mask_batch[seq_idx, ...],
                          extra_text='mask mode: %s keep prob: %s' % (MASK_MODES[FLAGS.mask_mode], FLAGS.keep_prob),
                          save_path=save_path, figwidth=figwidth, figheight=figheight)

    elif FLAGS.test_mode == "write_data":
        data_split = 'Validate'

        h5file = h5.File("%s_data_out_%d_%.1f.h5" %
                         (config.save_path,
                          FLAGS.mask_mode, FLAGS.keep_prob), "w")
        for _ in trange(val_batches):

            labs_batch, poses_batch, mask_batch, gen_inputs = get_inputs()

            gen_outputs = model_wrap.gen_model.predict(gen_inputs, config.batch_size)

            if config.normalize_data:
                poses_batch = data_input.denormalize_poses(poses_batch)
                gen_outputs = data_input.denormalize_poses(gen_outputs)

            for j in range(config.batch_size):
                seq_idx, subject, action, plen = labs_batch[j, ...]

                sub_array = np.array(subject + 1)
                act_array = np.array(action + 1)
                pose_array = gen_outputs[j, ...]
                pose_array = np.transpose(pose_array, (0, 2, 1))
                if config.data_set == 'NTURGBD':
                    pose_array = np.concatenate([pose_array, np.zeros_like(pose_array)])

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

    elif FLAGS.test_mode == "dmnn_score":
        FLAGS.config_file = None
        FLAGS.save_path = FLAGS.dmnn_save_path
        config = get_config(FLAGS)

        # Model building
        if config.model_type == 'dmnn':
            if config.model_version == 'v1':
                model_wrap_dmnn = DMNNv1(config)

        model_wrap_dmnn.model = restore_keras_model(model_wrap_dmnn.model, config.save_path + '_weights.hdf5')

        real_eval_sum = 0
        gen_eval_sum = 0
        bl_eval_sum = 0

        t = trange(val_batches)
        for i in t:

            labs_batch, poses_batch, mask_batch, gen_inputs = get_inputs()

            gen_outputs = model_wrap.gen_model.predict(gen_inputs, config.batch_size)

            real_loss, real_acc = model_wrap_dmnn.model.evaluate(poses_batch, labs_batch[:, 2], batch_size=config.batch_size, verbose=2)
            real_eval_sum += real_acc
            gen_loss, gen_acc = model_wrap_dmnn.model.evaluate(gen_outputs, labs_batch[:, 2], batch_size=config.batch_size, verbose=2)
            gen_eval_sum += gen_acc

            bl_batch = np.empty_like(poses_batch)
            for j in range(config.batch_size):
                bl_batch[j, ...] = constant_baseline(poses_batch[j, ...], mask_batch[j, ...])

            bl_loss, bl_acc = model_wrap_dmnn.model.evaluate(bl_batch, labs_batch[:, 2], batch_size=config.batch_size, verbose=2)
            bl_eval_sum += bl_acc

            t.set_postfix({'real_eval': real_eval_sum / (i + 1),
                           'gen_eval': gen_eval_sum / (i + 1),
                           'bl_eval': bl_eval_sum / (i + 1)})






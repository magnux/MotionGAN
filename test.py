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
flags.DEFINE_multi_string("model_path", None, "Model output directory")
flags.DEFINE_string("test_mode", "show_images", "Test modes: show_images, write_images, write_data, dmnn_score")
flags.DEFINE_string("dmnn_path", None, "Path to trained DMNN model")
flags.DEFINE_string("images_mode", "gif", "Image modes: gif, png")
flags.DEFINE_integer("mask_mode", 3, "Mask modes: 0:%s, 1:%s, 2:%s, 3:%s, 4:%s" % MASK_MODES)
flags.DEFINE_float("keep_prob", 0.8, "Probability of keeping input data. (1 == Keep All)")
FLAGS = flags.FLAGS

if __name__ == "__main__":
    # Config stuff
    batch_size = 1
    configs = []
    model_wraps = []
    # Hacks to fill undefined, but necessary flags
    tf.flags.DEFINE_string("config_file", None, None)
    tf.flags.DEFINE_string("save_path", None, None)
    for save_path in FLAGS.model_path:
        FLAGS.save_path = save_path
        config = get_config(FLAGS)
        config.only_val = True
        config.batch_size = batch_size

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

        configs.append(config)
        model_wraps.append(model_wrap)

    # TODO: assert all configs are for the same dataset

    if FLAGS.test_mode == "write_images":
        images_path = "%s_test_images_%s/" % \
                      ('_'.join(FLAGS.model_path), FLAGS.images_mode)
        if not tf.gfile.Exists(images_path):
            tf.gfile.MkDir(images_path)

    njoints = configs[0].njoints
    seq_len = model_wraps[0].seq_len
    body_members = configs[0].body_members

    data_input = DataInput(configs[0])
    val_batches = data_input.val_epoch_size
    val_generator = data_input.batch_generator(False)

    def gen_mask(mask_type=0, keep_prob=1.0):
        # Default mask, no mask
        mask = np.ones(shape=(batch_size, njoints, seq_len, 1))
        if mask_type == 1:  # Future Prediction
            mask[:, :, np.int(seq_len * keep_prob):, :] = 0.0
        elif mask_type == 2:  # Occlusion Simulation
            rand_joints = np.random.randint(njoints, size=np.int(njoints * (1.0 - keep_prob)))
            mask[:, rand_joints, :, :] = 0.0
        elif mask_type == 3:  # Structured Occlusion Simulation
            rand_joints = set()
            while ((njoints - len(rand_joints)) >
                   (njoints * keep_prob)):
                joints_to_add = (body_members.values()[np.random.randint(len(body_members))])['joints']
                for joint in joints_to_add:
                    rand_joints.add(joint)
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

        if configs[0].latent_cond_dim > 0:
            latent_noise = np.random.uniform(
                size=(batch_size, configs[0].latent_cond_dim))
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

    if "images" in FLAGS.test_mode:

        for i in trange(val_batches):
            labs_batch, poses_batch, mask_batch, gen_inputs = get_inputs()

            gen_outputs = []
            for model_wrap in model_wraps:
                gen_outputs.append(model_wrap.gen_model.predict(gen_inputs, batch_size))

            if configs[0].normalize_data:
                poses_batch = data_input.denormalize_poses(poses_batch)
                for j in range(len(gen_outputs)):
                    gen_outputs[j] = data_input.denormalize_poses(gen_outputs[j])

            # rand_indices = np.random.permutation(batch_size)

            for j in range(batch_size):
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

                plot_func(np.concatenate([poses_batch[np.newaxis, seq_idx, ...]] +
                                         [gen_output[np.newaxis, seq_idx, ...] for gen_output in gen_outputs] +
                                         [constant_seq]),
                          labs_batch[seq_idx, ...],
                          configs[0].data_set,
                          seq_masks=mask_batch[seq_idx, ...],
                          extra_text='mask mode: %s keep prob: %s' % (MASK_MODES[FLAGS.mask_mode], FLAGS.keep_prob),
                          save_path=save_path, figwidth=figwidth, figheight=figheight)

    elif FLAGS.test_mode == "write_data":
        data_split = 'Validate'

        h5files = []
        for config in configs:
            h5files.append(h5.File("%s_data_out_%d_%.1f.h5" %
                                   (config.save_path, FLAGS.mask_mode, FLAGS.keep_prob), "w"))

        for _ in trange(val_batches):

            labs_batch, poses_batch, mask_batch, gen_inputs = get_inputs()

            for m, model_wrap in enumerate(model_wraps):
                gen_outputs = model_wrap.gen_model.predict(gen_inputs, batch_size)

                if configs[m].normalize_data:
                    gen_outputs = data_input.denormalize_poses(gen_outputs)

                for j in range(batch_size):
                    seq_idx, subject, action, plen = labs_batch[j, ...]

                    sub_array = np.array(subject + 1)
                    act_array = np.array(action + 1)
                    pose_array = gen_outputs[j, ...]
                    pose_array = np.transpose(pose_array, (0, 2, 1))
                    if config.data_set == 'NTURGBD':
                        pose_array = np.concatenate([pose_array, np.zeros_like(pose_array)])

                    data_path = '%s/%s/SEQ%d/' % (model_wrap.data_set, data_split, seq_idx)
                    h5files[m].create_dataset(
                        data_path + 'Subject', np.shape(sub_array),
                        dtype='int32', data=sub_array
                    )
                    h5files[m].create_dataset(
                        data_path + 'Action', np.shape(act_array),
                        dtype='int32', data=act_array
                    )
                    h5files[m].create_dataset(
                        data_path + 'Pose', np.shape(pose_array),
                        dtype='float32', data=pose_array
                    )

        for h5file in h5files:
            h5file.flush()
            h5file.close()

    elif FLAGS.test_mode == "dmnn_score":
        FLAGS.save_path = FLAGS.dmnn_path
        config = get_config(FLAGS)
        config.batch_size = batch_size

        # Model building
        if config.model_type == 'dmnn':
            if config.model_version == 'v1':
                model_wrap_dmnn = DMNNv1(config)

        model_wrap_dmnn.model = restore_keras_model(model_wrap_dmnn.model, config.save_path + '_weights.hdf5')

        accs = {'real_acc': 0, 'bl_acc': 0}

        for model_wrap in model_wraps:
            accs[model_wrap.name + '_acc'] = 0

        t = trange(val_batches)
        for i in t:

            labs_batch, poses_batch, mask_batch, gen_inputs = get_inputs()

            for model_wrap in model_wraps:
                gen_outputs = model_wrap.gen_model.predict(gen_inputs, batch_size)
                if configs[0].normalize_data:
                    gen_outputs = data_input.denormalize_poses(gen_outputs)
                gen_loss, gen_acc = model_wrap_dmnn.model.evaluate(gen_outputs, labs_batch[:, 2], batch_size=batch_size, verbose=2)
                accs[model_wrap.name + '_acc'] += gen_acc

            if configs[0].normalize_data:
                poses_batch = data_input.denormalize_poses(poses_batch)

            real_loss, real_acc = model_wrap_dmnn.model.evaluate(poses_batch, labs_batch[:, 2], batch_size=batch_size, verbose=2)
            accs['real_acc'] += real_acc

            bl_batch = np.empty_like(poses_batch)
            for j in range(batch_size):
                bl_batch[j, ...] = constant_baseline(poses_batch[j, ...], mask_batch[j, ...])

            bl_loss, bl_acc = model_wrap_dmnn.model.evaluate(bl_batch, labs_batch[:, 2], batch_size=batch_size, verbose=2)
            accs['bl_acc'] += bl_acc

            mean_accs = {}
            for key, value in accs.items():
                mean_accs[key] = value / (i + 1)

            t.set_postfix(mean_accs)






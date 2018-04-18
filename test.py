from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from config import get_config
from data_input import DataInput
from models.motiongan import MotionGANV1, MotionGANV2, MotionGANV3, MotionGANV4
from models.dmnn import DMNNv1
from utils.restore_keras_model import restore_keras_model
from utils.viz import plot_seq_gif, plot_seq_pano
from utils.baselines import constant_baseline, burke_baseline
import h5py as h5
from tqdm import trange
from collections import OrderedDict

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
    batch_size = 1 if not "dmnn_score" in FLAGS.test_mode else 256
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
    # TODO: assert all configs have the same remove_hip
    data_input = DataInput(configs[0])
    val_batches = data_input.val_epoch_size
    val_generator = data_input.batch_generator(False)

    if FLAGS.test_mode == "write_images":
        images_path = "%s_test_images_%s/" % \
                      ('_'.join(FLAGS.model_path), FLAGS.images_mode)
        if not tf.gfile.Exists(images_path):
            tf.gfile.MkDir(images_path)

    njoints = configs[0].njoints
    seq_len = model_wraps[0].seq_len
    body_members = configs[0].body_members

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

        # This unmasks first frame for all sequences
        mask[:, :, [0, -1], :] = 1.0
        return mask

    def get_inputs():
        labs_batch, poses_batch, hip_poses_batch = val_generator.next()

        mask_batch = poses_batch[..., 3, np.newaxis]
        mask_batch = mask_batch * gen_mask(FLAGS.mask_mode, FLAGS.keep_prob)
        poses_batch = poses_batch[..., :3]
        hip_mask_batch = hip_poses_batch[..., 3, np.newaxis]
        hip_poses_batch = hip_poses_batch[..., :3]

        gen_inputs = [poses_batch, mask_batch]

        if configs[0].latent_cond_dim > 0:
            latent_noise = np.random.uniform(
                size=(batch_size, configs[0].latent_cond_dim))
            gen_inputs.append(latent_noise)

        return labs_batch, poses_batch, hip_poses_batch, mask_batch, hip_mask_batch, gen_inputs

    if "images" in FLAGS.test_mode:

        for i in trange(val_batches):
            labs_batch, poses_batch, hip_poses_batch, mask_batch, hip_mask_batch, gen_inputs = get_inputs()

            gen_outputs = []
            for m, model_wrap in enumerate(model_wraps):
                gen_outputs.append(model_wrap.gen_model.predict(gen_inputs, batch_size))

                if configs[m].normalize_data:
                    gen_outputs = data_input.unnormalize_poses(gen_outputs)

                if configs[m].remove_hip:
                    gen_outputs = np.concatenate([hip_poses_batch, gen_outputs + hip_poses_batch], axis=1)

            if configs[0].normalize_data:
                poses_batch = data_input.unnormalize_poses(poses_batch)

            if configs[0].remove_hip:
                poses_batch = np.concatenate([hip_poses_batch, poses_batch + hip_poses_batch], axis=1)
                mask_batch = np.concatenate([hip_mask_batch, mask_batch], axis=1)

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
                burke_seq = \
                    burke_baseline(poses_batch[seq_idx, ...], mask_batch[seq_idx, ...])
                burke_seq = np.expand_dims(burke_seq, 0)

                plot_func(np.concatenate([poses_batch[np.newaxis, seq_idx, ...], constant_seq, burke_seq] +
                                         [gen_output[np.newaxis, seq_idx, ...] for gen_output in gen_outputs]),
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

            labs_batch, poses_batch, hip_poses_batch, mask_batch, hip_mask_batch, gen_inputs = get_inputs()

            for m, model_wrap in enumerate(model_wraps):
                gen_outputs = model_wrap.gen_model.predict(gen_inputs, batch_size)

                if configs[m].normalize_data:
                    gen_outputs = data_input.unnormalize_poses(gen_outputs)

                if configs[m].remove_hip:
                    gen_outputs = np.concatenate([hip_poses_batch, gen_outputs + hip_poses_batch], axis=1)

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

    elif "dmnn_score" in FLAGS.test_mode:

        if FLAGS.dmnn_path is not None:
            FLAGS.save_path = FLAGS.dmnn_path
            config = get_config(FLAGS)
            config.batch_size = batch_size

            # Model building
            if config.model_type == 'dmnn':
                if config.model_version == 'v1':
                    model_wrap_dmnn = DMNNv1(config)

            model_wrap_dmnn.model = restore_keras_model(model_wrap_dmnn.model, config.save_path + '_weights.hdf5')

        def run_dmnn_score():

            accs = OrderedDict({'real_acc': 0, 'const_acc': 0, 'burke_acc': 0})
            p2ps = OrderedDict({'const_p2p': 0, 'burke_p2p': 0})
            dms = OrderedDict({'const_dm': 0, 'burke_dm': 0})

            for m in range(len(model_wraps)):
                accs[FLAGS.model_path[m] + '_acc'] = 0
                p2ps[FLAGS.model_path[m] + '_p2p'] = 0
                dms[FLAGS.model_path[m] + '_dm'] = 0

            def unnormalize_batch(batch, hip_batch, m=0):
                if configs[m].normalize_data:
                    batch = data_input.unnormalize_poses(batch)
                if configs[m].remove_hip:
                    batch = np.concatenate([hip_batch, batch + hip_batch], axis=1)
                return batch

            def p2pd(x, y):
                return np.mean(np.sqrt(np.sum(np.square(x - y), axis=-1)))

            def edm(x, y=None):
                y = x if y is None else y
                x = np.expand_dims(x, axis=1)
                y = np.expand_dims(y, axis=2)
                return np.sqrt(np.sum(np.square(x - y), axis=-1))

            t = trange(val_batches)
            for i in t:

                labs_batch, poses_batch, hip_poses_batch, mask_batch, hip_mask_batch, gen_inputs = get_inputs()

                re_poses_batch = unnormalize_batch(poses_batch, hip_poses_batch)
                re_poses_batch_edm = edm(re_poses_batch)

                for m, model_wrap in enumerate(model_wraps):
                    gen_outputs = model_wrap.gen_model.predict(gen_inputs, batch_size)
                    if FLAGS.dmnn_path is not None:
                        _, gen_acc = model_wrap_dmnn.model.evaluate(gen_outputs, labs_batch[:, 2], batch_size=batch_size, verbose=2)
                        accs[FLAGS.model_path[m] + '_acc'] += gen_acc

                    gen_outputs = unnormalize_batch(gen_outputs, hip_poses_batch, m)
                    p2ps[FLAGS.model_path[m] + '_p2p'] += p2pd(re_poses_batch, gen_outputs)
                    dms[FLAGS.model_path[m] + '_dm'] += np.mean(np.abs(re_poses_batch_edm - edm(gen_outputs)))

                if FLAGS.dmnn_path is not None:
                    _, real_acc = model_wrap_dmnn.model.evaluate(poses_batch, labs_batch[:, 2], batch_size=batch_size, verbose=2)
                    accs['real_acc'] += real_acc

                constant_batch = np.empty_like(poses_batch)
                burke_batch = np.empty_like(poses_batch)
                for j in range(batch_size):
                    constant_batch[j, ...] = constant_baseline(poses_batch[j, ...], mask_batch[j, ...])
                    burke_batch[j, ...] = burke_baseline(poses_batch[j, ...], mask_batch[j, ...])

                if FLAGS.dmnn_path is not None:
                    _, const_acc = model_wrap_dmnn.model.evaluate(constant_batch, labs_batch[:, 2], batch_size=batch_size, verbose=2)
                    accs['const_acc'] += const_acc

                constant_batch = unnormalize_batch(constant_batch, hip_poses_batch)
                p2ps['const_p2p'] += p2pd(re_poses_batch, constant_batch)
                dms['const_dm'] += np.mean(np.abs(re_poses_batch_edm - edm(constant_batch)))

                if FLAGS.dmnn_path is not None:
                    _, burke_acc = model_wrap_dmnn.model.evaluate(burke_batch, labs_batch[:, 2], batch_size=batch_size, verbose=2)
                    accs['burke_acc'] += burke_acc

                burke_batch = unnormalize_batch(burke_batch, hip_poses_batch)
                p2ps['burke_p2p'] += p2pd(re_poses_batch, burke_batch)
                dms['burke_dm'] += np.mean(np.abs(re_poses_batch_edm - edm(burke_batch)))

                mean_accs = {}
                for key, value in accs.items():
                    mean_accs[key] = value / (i + 1)

                t.set_postfix(mean_accs)

            def make_mean(my_dict):
                for key, value in my_dict.items():
                    my_dict[key] = value / val_batches
                return my_dict

            return make_mean(accs), make_mean(p2ps), make_mean(dms)

        if FLAGS.test_mode == "dmnn_score_table":

            PROBS = np.arange(0.0, 1.1, 0.1)

            for m in range(1, len(MASK_MODES)):
                accs_table = np.zeros((len(PROBS), len(model_wraps) + 3))
                p2ps_table = np.zeros((len(PROBS), len(model_wraps) + 2))
                dms_table = np.zeros((len(PROBS), len(model_wraps) + 2))
                for p, prob in enumerate(PROBS):
                    FLAGS.mask_mode = m
                    FLAGS.keep_prob = prob

                    accs, p2ps, dms = run_dmnn_score()
                    accs_table[p, :] = accs.values()
                    p2ps_table[p, :] = p2ps.values()
                    dms_table[p, :] = dms.values()

                np.savetxt('save/test_accs_%d.txt' % m, accs_table, '%.8e', ',', '\n', ','.join(accs.keys()))
                np.savetxt('save/test_p2ps_%d.txt' % m, p2ps_table, '%.8e', ',', '\n', ','.join(p2ps.keys()))
                np.savetxt('save/test_dms_%d.txt' % m, dms_table, '%.8e', ',', '\n', ','.join(dms.keys()))

        else:
            run_dmnn_score()








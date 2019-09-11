from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import scipy as sp
from scipy import signal
from config import get_config
from data_input import DataInput
from models.motiongan import get_model
from models.dmnn import DMNNv1
from utils.restore_keras_model import restore_keras_model
from utils.viz import plot_seq_gif, plot_seq_pano, plot_seq_frozen
from utils.seq_utils import MASK_MODES, gen_mask, linear_baseline, burke_baseline, post_process, seq_to_angles_transformer, get_angles_mask, gen_latent_noise, _some_variables, fkl, rotate_start
import h5py as h5
from tqdm import trange
from collections import OrderedDict
from colorama import Fore, Back, Style
import utils.npangles as npangles

logging = tf.logging
flags = tf.flags
flags.DEFINE_bool("verbose", False, "To talk or not to talk")
flags.DEFINE_multi_string("model_path", None, "Model output directory")
flags.DEFINE_string("test_mode", "show_images", "Test modes: show_images, write_images, write_data, dmnn_score, dmnn_score_table, hmp_l2_comp, paper_metrics")
flags.DEFINE_string("dmnn_path", None, "Path to trained DMNN model")
flags.DEFINE_string("images_mode", "gif", "Image modes: gif, png")
flags.DEFINE_integer("mask_mode", 1, "Mask modes: " + ' '.join(['%d:%s' % tup for tup in enumerate(MASK_MODES)]))
flags.DEFINE_float("keep_prob", 0.5, "Probability of keeping input data. (1 == Keep All)")
FLAGS = flags.FLAGS


def _reset_rand_seed(seed=42):
    np.random.seed(seed)
    tf.set_random_seed(seed)


if __name__ == "__main__":
    _reset_rand_seed()
    # Config stuff
    batch_size = 1
    if "dmnn_score" in FLAGS.test_mode or \
        "paper_metrics" in FLAGS.test_mode or \
        "alternate_seq_dist" in FLAGS.test_mode:
        batch_size = 128
    elif "plot_survey" in FLAGS.test_mode:
        batch_size = 120

    configs = []
    model_wraps = []
    # Hacks to fill undefined, but necessary flags
    tf.flags.DEFINE_string("config_file", None, None)
    tf.flags.DEFINE_string("save_path", None, None)

    for save_path in FLAGS.model_path:
        FLAGS.save_path = save_path
        config = get_config(FLAGS)
        config.only_val = True if "paper_metrics" not in FLAGS.test_mode else False
        config.batch_size = batch_size

        # Model building
        if config.model_type == 'motiongan':
            model_wrap = get_model(config)

        if FLAGS.verbose:
            print('Discriminator model:')
            print(model_wrap.disc_model.summary())
            print(len(model_wrap.disc_model.layers))
            print('Generator model:')
            print(model_wrap.gen_model.summary())
            print(len(model_wrap.gen_model.layers))
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
    if "paper_metrics" in FLAGS.test_mode:
        configs[0].crop_len = (configs[0].crop_len // 2) + configs[0].crop_len
        configs[0].pick_num = (configs[0].pick_num // 2) + configs[0].pick_num
    data_input = DataInput(configs[0])
    _reset_rand_seed()
    train_batches = data_input.train_epoch_size
    train_generator = data_input.batch_generator(True)
    val_batches = data_input.val_epoch_size
    val_generator = data_input.batch_generator(False)

    # if FLAGS.test_mode == "write_images" or FLAGS.test_mode == "plot_survey":
    images_path = "%s_test_images_%s/" % \
                  (configs[0].save_path, FLAGS.images_mode)
    if not tf.gfile.Exists(images_path):
        tf.gfile.MkDir(images_path)
    if "plot_survey" in FLAGS.test_mode:
        for s in range(4):
            surv_img_path = images_path + ("survey_%d/" % s)
            if not tf.gfile.Exists(surv_img_path):
                tf.gfile.MkDir(surv_img_path)
            for g in range(2):
                gt_img_path = surv_img_path + ("gt_%d/" % g)
                if not tf.gfile.Exists(gt_img_path):
                    tf.gfile.MkDir(gt_img_path)
    if "rebuttal" in FLAGS.test_mode:
        reb_img_path = images_path + "rebuttal/"
        if not tf.gfile.Exists(reb_img_path):
            tf.gfile.MkDir(reb_img_path)

    njoints = configs[0].njoints
    seq_len = model_wraps[0].seq_len
    body_members = configs[0].body_members  # if not configs[0].data_set == 'Human36' else configs[0].full_body_members
    angle_trans = seq_to_angles_transformer(body_members)

    def get_inputs(baseline_mode=False):
        labs_batch, poses_batch = val_generator.next()

        mask_batch = poses_batch[..., 3, np.newaxis]
        mask_batch = mask_batch * gen_mask(FLAGS.mask_mode, FLAGS.keep_prob,
                                           batch_size, njoints, seq_len, body_members, baseline_mode)
        poses_batch = poses_batch[..., :3]

        return labs_batch, poses_batch, mask_batch

    if "images" in FLAGS.test_mode:

        for i in trange(val_batches):
            labs_batch, poses_batch, mask_batch = get_inputs(False)
            labels = np.reshape(labs_batch[:, 2], (batch_size, 1))

            gen_outputs = []
            # proc_gen_outputs = []
            for m, model_wrap in enumerate(model_wraps):
                gen_inputs = [poses_batch, mask_batch]
                if configs[m].action_cond:
                    gen_inputs.append(labels)
                if configs[m].latent_cond_dim > 0:
                    latent_noise = gen_latent_noise(batch_size, configs[m].latent_cond_dim)
                    gen_inputs.append(latent_noise)
                gen_output = model_wrap.gen_model.predict(gen_inputs, batch_size)
                # proc_gen_output = np.empty_like(gen_output)
                # for j in range(batch_size):
                #     proc_gen_output[j, ...] = post_process(poses_batch[j, ...], gen_output[j, ...],
                #                                       mask_batch[j, ...], body_members)
                if configs[m].normalize_data:
                    gen_output = data_input.unnormalize_poses(gen_output)
                    # proc_gen_output = data_input.unnormalize_poses(proc_gen_output)
                gen_outputs.append(gen_output)
                # proc_gen_outputs.append(proc_gen_output)

            if configs[0].normalize_data:
                poses_batch = data_input.unnormalize_poses(poses_batch)

            # rand_indices = np.random.permutation(batch_size)

            for j in range(batch_size):
                # seq_idx = rand_indices[j]
                seq_idx = j

                save_path = None
                if FLAGS.test_mode == "write_images":
                    save_path = images_path + ("%d_%d.%s" % (i, j, FLAGS.images_mode))
                    np.save(images_path + ("%d_%d_gt.npy" % (i, j)), poses_batch[np.newaxis, seq_idx, ...])
                    np.save(images_path + ("%d_%d_gen.npy" % (i, j)), gen_output[np.newaxis, seq_idx, ...])

                if FLAGS.images_mode == "gif":
                    plot_func = plot_seq_gif
                    figwidth = 256 * (len(configs) + 1)
                    figheight = 256
                elif FLAGS.images_mode == "png":
                    plot_func = plot_seq_frozen  # plot_seq_pano
                    figwidth = 768
                    figheight = 256 * (len(configs) + 1)

                # linear_seq =\
                #     linear_baseline(poses_batch[seq_idx, ...], mask_batch[seq_idx, ...])
                # linear_seq = np.expand_dims(linear_seq, 0)
                # burke_seq = \
                #     burke_baseline(poses_batch[seq_idx, ...], mask_batch[seq_idx, ...])
                # burke_seq = np.expand_dims(burke_seq, 0)

                plot_func(np.concatenate([poses_batch[np.newaxis, seq_idx, ...]] + # [poses_batch[np.newaxis, seq_idx, ...], linear_seq, burke_seq]
                                         [gen_output[np.newaxis, seq_idx, ...] for gen_output in gen_outputs] ) # +
                                         , #  [proc_gen_output[np.newaxis, seq_idx, ...] for proc_gen_output in proc_gen_outputs])
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

            labs_batch, poses_batch, mask_batch = get_inputs(False)

            for m, model_wrap in enumerate(model_wraps):
                gen_inputs = [poses_batch, mask_batch]
                if configs[m].action_cond:
                    labels = np.reshape(labs_batch[:, 2], (batch_size, 1))
                    gen_inputs.append(labels)
                if configs[m].latent_cond_dim > 0:
                    latent_noise = gen_latent_noise(batch_size, configs[m].latent_cond_dim)
                    gen_inputs.append(latent_noise)
                gen_output = model_wrap.gen_model.predict(gen_inputs, batch_size)
                for j in range(batch_size):
                    gen_output[j, ...] = post_process(poses_batch[j, ...], gen_output[j, ...],
                                                      mask_batch[j, ...], body_members)
                if configs[m].normalize_data:
                    gen_output = data_input.unnormalize_poses(gen_output)

                for j in range(batch_size):
                    seq_idx, subject, action, plen = labs_batch[j, ...]

                    sub_array = np.array(subject + 1)
                    act_array = np.array(action + 1)
                    pose_array = gen_output[j, ...]
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

            accs = OrderedDict({'real_acc': 0, 'linear_acc': 0, 'burke_acc': 0})
            p2ps = OrderedDict({'linear_p2p': 0, 'burke_p2p': 0})
            dms = OrderedDict({'linear_dm': 0, 'burke_dm': 0})
            angles = OrderedDict({'linear_angle': 0, 'burke_angle': 0})

            for m in range(len(model_wraps)):
                accs[FLAGS.model_path[m] + '_acc'] = 0
                p2ps[FLAGS.model_path[m] + '_p2p'] = 0
                dms[FLAGS.model_path[m] + '_dm'] = 0
                angles[FLAGS.model_path[m] + '_angle'] = 0

            def unnormalize_batch(batch, m=0):
                if configs[m].normalize_data:
                    batch = data_input.unnormalize_poses(batch)
                return batch

            def p2pd(x, y):
                return np.sqrt(np.sum(np.square(x - y), axis=-1, keepdims=True))

            def edm(x, y=None):
                y = x if y is None else y
                x = np.expand_dims(x, axis=1)
                y = np.expand_dims(y, axis=2)
                return np.sqrt(np.sum(np.square(x - y), axis=-1, keepdims=True))

            t = trange(val_batches)
            for i in t:

                labs_batch, poses_batch, mask_batch = get_inputs(True)

                unorm_poses_batch = unnormalize_batch(poses_batch)
                unorm_poses_batch_edm = edm(unorm_poses_batch)
                unorm_poses_batch_angles = angle_trans(unorm_poses_batch)

                p2ps_occ_num = np.sum(1.0 - mask_batch) + 1e-8
                dms_mask_batch = np.expand_dims(mask_batch, axis=1) * np.expand_dims(mask_batch, axis=2)
                dms_occ_num = np.sum(1.0 - dms_mask_batch) + 1e-8
                angles_mask_batch = get_angles_mask(mask_batch, body_members)
                angles_occ_num = np.sum(1.0 - angles_mask_batch) + 1e-8

                for m, model_wrap in enumerate(model_wraps):
                    gen_inputs = [poses_batch, mask_batch]
                    if configs[m].action_cond:
                        labels = np.reshape(labs_batch[:, 2], (batch_size, 1))
                        gen_inputs.append(labels)
                    if configs[m].latent_cond_dim > 0:
                        latent_noise = gen_latent_noise(batch_size, configs[m].latent_cond_dim)
                        gen_inputs.append(latent_noise)
                    gen_output = model_wrap.gen_model.predict(gen_inputs, batch_size)
                    # for j in range(batch_size):
                    #     gen_output[j, ...] = post_process(poses_batch[j, ...], gen_output[j, ...],
                    #                                       mask_batch[j, ...], body_members)
                    if FLAGS.dmnn_path is not None:
                        _, gen_acc = model_wrap_dmnn.model.evaluate(gen_output, labs_batch[:, 2], batch_size=batch_size, verbose=2)
                        accs[FLAGS.model_path[m] + '_acc'] += gen_acc

                    gen_output = unnormalize_batch(gen_output, m)
                    p2ps[FLAGS.model_path[m] + '_p2p'] += np.sum(p2pd(unorm_poses_batch, gen_output) * (1.0 - mask_batch)) / p2ps_occ_num
                    dms[FLAGS.model_path[m] + '_dm'] += np.sum(np.abs(unorm_poses_batch_edm - edm(gen_output)) * (1.0 - dms_mask_batch)) / dms_occ_num
                    angles[FLAGS.model_path[m] + '_angle'] += np.sum(p2pd(unorm_poses_batch_angles, angle_trans(gen_output)) * (1.0 - angles_mask_batch)) / angles_occ_num

                if FLAGS.dmnn_path is not None:
                    _, real_acc = model_wrap_dmnn.model.evaluate(poses_batch, labs_batch[:, 2], batch_size=batch_size, verbose=2)
                    accs['real_acc'] += real_acc

                linear_batch = np.empty_like(poses_batch)
                burke_batch = np.empty_like(poses_batch)
                for j in range(batch_size):
                    linear_batch[j, ...] = linear_baseline(poses_batch[j, ...], mask_batch[j, ...])
                    burke_batch[j, ...] = burke_baseline(poses_batch[j, ...], mask_batch[j, ...])

                if FLAGS.dmnn_path is not None:
                    _, linear_acc = model_wrap_dmnn.model.evaluate(linear_batch, labs_batch[:, 2], batch_size=batch_size, verbose=2)
                    accs['linear_acc'] += linear_acc

                linear_batch = unnormalize_batch(linear_batch)
                p2ps['linear_p2p'] += np.sum(p2pd(unorm_poses_batch, linear_batch) * (1.0 - mask_batch)) / p2ps_occ_num
                dms['linear_dm'] += np.sum(np.abs(unorm_poses_batch_edm - edm(linear_batch)) * (1.0 - dms_mask_batch)) / dms_occ_num
                angles['linear_angle'] += np.sum(p2pd(unorm_poses_batch_angles, angle_trans(linear_batch)) * (1.0 - angles_mask_batch)) / angles_occ_num

                if FLAGS.dmnn_path is not None:
                    _, burke_acc = model_wrap_dmnn.model.evaluate(burke_batch, labs_batch[:, 2], batch_size=batch_size, verbose=2)
                    accs['burke_acc'] += burke_acc

                burke_batch = unnormalize_batch(burke_batch)
                p2ps['burke_p2p'] += np.sum(p2pd(unorm_poses_batch, burke_batch) * (1.0 - mask_batch)) / p2ps_occ_num
                dms['burke_dm'] += np.sum(np.abs(unorm_poses_batch_edm - edm(burke_batch)) * (1.0 - dms_mask_batch)) / dms_occ_num
                angles['burke_angle'] += np.sum(p2pd(unorm_poses_batch_angles, angle_trans(burke_batch)) * (1.0 - angles_mask_batch)) / angles_occ_num

                mean_accs = {}
                for key, value in accs.items():
                    mean_accs[key] = value / (i + 1)

                t.set_postfix(mean_accs)

            def make_mean(my_dict):
                for key, value in my_dict.items():
                    my_dict[key] = value / val_batches
                return my_dict

            return make_mean(accs), make_mean(p2ps), make_mean(dms), make_mean(angles)

        if FLAGS.test_mode == "dmnn_score_table":

            # PROBS = np.arange(0.0, 1.1, 0.1)
            PROBS = [0.2]

            for m in range(1, len(MASK_MODES)):
                accs_table = np.zeros((len(PROBS), len(model_wraps) + 3))
                p2ps_table = np.zeros((len(PROBS), len(model_wraps) + 2))
                dms_table = np.zeros((len(PROBS), len(model_wraps) + 2))
                angles_table = np.zeros((len(PROBS), len(model_wraps) + 2))
                for p, prob in enumerate(PROBS):
                    FLAGS.mask_mode = m
                    FLAGS.keep_prob = prob

                    accs, p2ps, dms, angles = run_dmnn_score()
                    accs_table[p, :] = accs.values()
                    p2ps_table[p, :] = p2ps.values()
                    dms_table[p, :] = dms.values()
                    angles_table[p, :] = angles.values()

                np.savetxt('save/test_accs_%d.txt' % m, accs_table, '%.8e', ',', '\n', ','.join(accs.keys()))
                np.savetxt('save/test_p2ps_%d.txt' % m, p2ps_table, '%.8e', ',', '\n', ','.join(p2ps.keys()))
                np.savetxt('save/test_dms_%d.txt' % m, dms_table, '%.8e', ',', '\n', ','.join(dms.keys()))
                np.savetxt('save/test_angles_%d.txt' % m, angles_table, '%.8e', ',', '\n', ','.join(angles.keys()))

        else:
            run_dmnn_score()
    elif FLAGS.test_mode == "hmp_l2_comp":
        from utils.human36_expmaps_to_h5 import actions

        def em2eul(a):
            return npangles.rotmat_to_euler(npangles.expmap_to_rotmat(a))

        def euc_error(x, y):
            x = np.reshape(x, (x.shape[0], -1))
            y = np.reshape(y, (y.shape[0], -1))
            return np.sqrt(np.sum(np.square(x - y), 1))

        def motion_error(x, y):
            return euc_error(x[1:, :] - x[:-1, :], y[1:, :] - y[:-1, :])

        def subsample(seq):
            return seq[range(0, int(seq.shape[0]), 5), :]

        h36_coords_used_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
        parent, offset, rotInd, expmapInd = _some_variables()

        def to_coords(seq_angles):
            seq_coords = np.empty((1, len(h36_coords_used_joints), seq_angles.shape[0], 3))
            for i in range(seq_angles.shape[0]):
                frame_coords = fkl(seq_angles[i, :], parent, offset, rotInd, expmapInd)
                seq_coords[0, :, i, :] = frame_coords[h36_coords_used_joints, :]
            seq_coords[..., 1] = seq_coords[..., 1] * -1  # Inverting y axis for visualization purposes
            return seq_coords

        def edm(x, y=None):
            y = x if y is None else y
            x = np.expand_dims(x, axis=1)
            y = np.expand_dims(y, axis=2)
            return np.sqrt(np.sum(np.square(x - y), axis=-1))

        def flat_edm(x):
            idxs = np.triu_indices(x.shape[1], k=1)
            x_edm = edm(x)
            x_edm = x_edm[:, idxs[0], idxs[1], :]
            x_edm = np.transpose(np.squeeze(x_edm, 0), (1, 0))
            return x_edm

        with h5.File('../human-motion-prediction/samples.h5', "r") as sample_file:
            for act_idx, action in enumerate(actions):
                pred_len = seq_len // 2
                mean_errors_hmp = np.zeros((8, pred_len))
                mean_errors_mg = np.zeros((8, pred_len))
                for i in np.arange(8):
                    encoder_inputs = np.array(sample_file['expmap/encoder_inputs/{1}_{0}'.format(i, action)], dtype=np.float32)
                    decoder_inputs = np.array(sample_file['expmap/decoder_inputs/{1}_{0}'.format(i, action)], dtype=np.float32)
                    decoder_outputs = np.array(sample_file['expmap/decoder_outputs/{1}_{0}'.format(i, action)], dtype=np.float32)
                    input_seeds_sact = np.int32(sample_file['expmap/input_seeds_sact/{1}_{0}'.format(i, action)])
                    input_seeds_idx = np.int32(sample_file['expmap/input_seeds_idx/{1}_{0}'.format(i, action)])
                    input_seeds_seqlen = np.int32(sample_file['expmap/input_seeds_seqlen/{1}_{0}'.format(i, action)])

                    # print(input_seeds_sact, input_seeds_idx)

                    expmap_gt = np.array(sample_file['expmap/gt/{1}_{0}'.format(i, action)], dtype=np.float32)
                    if 'expmaps' not in configs[0].data_set:
                        expmap_gt = expmap_gt[4:, ...]  # Our model predicts every 200ms, first frames are not compared
                        expmap_gt = subsample(expmap_gt)
                    expmap_gt = expmap_gt[:pred_len, ...]

                    expmap_hmp = np.array(sample_file['expmap/preds/{1}_{0}'.format(i, action)], dtype=np.float32)
                    if 'expmaps' not in configs[0].data_set:
                        expmap_hmp = expmap_hmp[4:, ...]
                        expmap_hmp = subsample(expmap_hmp)
                    expmap_hmp = expmap_hmp[:pred_len, ...]

                    poses_batch = None
                    if 'expmaps' in configs[0].data_set:
                        poses_batch = np.concatenate([encoder_inputs, decoder_inputs[np.newaxis, 0, :], decoder_outputs], axis=0)
                        # poses_batch = subsample(poses_batch)
                        poses_batch = poses_batch[50 - pred_len:50 + pred_len, :]
                        poses_batch = np.transpose(np.reshape(poses_batch, (1, pred_len*2, 33, 3)), (0, 2, 1, 3))
                        poses_batch = poses_batch[:, configs[0].used_joints, :, :]
                    else:
                        for key in data_input.val_keys:
                            if np.int32(data_input.h5file[key + '/Action']) - 1 == act_idx:
                                pose = np.array(data_input.h5file[key + '/Pose'], dtype=np.float32)
                                pose, plen = data_input.process_pose(pose)
                                if np.ceil(plen / 2) == input_seeds_seqlen:
                                    pose = pose[:, input_seeds_idx:input_seeds_idx+200, :]
                                    pose = pose[:, range(0, 200, 10), :]
                                    poses_batch = np.reshape(pose, [batch_size] + data_input.pshape)
                                    poses_batch = poses_batch[..., :3]
                                    break

                    mask_batch = np.ones((1, njoints, pred_len*2, 1), dtype=np.float32)
                    mask_batch[:, :, pred_len:, :] = 0.0

                    if configs[0].normalize_data:
                        poses_batch = data_input.normalize_poses(poses_batch)

                    gen_inputs = [poses_batch, mask_batch]
                    if configs[0].action_cond:
                        action_label = np.ones((batch_size, 1), dtype=np.float32) * act_idx
                        gen_inputs.append(action_label)
                    if configs[0].latent_cond_dim > 0:
                        # latent_noise = gen_latent_noise(batch_size, configs[0].latent_cond_dim)
                        latent_noise = np.ones((batch_size, configs[0].latent_cond_dim), dtype=np.float32) * 0.5
                        gen_inputs.append(latent_noise)

                    gen_output = model_wrap.gen_model.predict(gen_inputs, batch_size)
                    # gen_output = np.tile(poses_batch[:, :, 9, np.newaxis, :], (1, 1, 20, 1))
                    # gen_output /= 2.0

                    # print(np.mean(np.abs(poses_batch[:, :, :pred_len, ...] - gen_output[:, :, :pred_len, ...])),
                    #       np.mean(np.abs(poses_batch[:, :,  pred_len:, ...] - gen_output[:, :,  pred_len:, ...])))

                    if configs[0].normalize_data:
                        gen_output = data_input.unnormalize_poses(gen_output)
                        poses_batch = data_input.unnormalize_poses(poses_batch)

                    # print(np.mean(np.abs(poses_batch[:, :, :pred_len, ...] - gen_output[:, :, :pred_len, ...])),
                    #       np.mean(np.abs(poses_batch[:, :,  pred_len:, ...] - gen_output[:, :,  pred_len:, ...])))

                    if 'expmaps' in configs[0].data_set:
                        expmap_mg = np.zeros((batch_size, configs[0].full_njoints, pred_len * 2, 3))
                        expmap_mg[:, configs[0].used_joints, :, :] = gen_output
                        expmap_pb = np.zeros((batch_size, configs[0].full_njoints, pred_len * 2, 3))
                        expmap_pb[:, configs[0].used_joints, :, :] = poses_batch
                    else:
                        expmap_mg = angle_trans(gen_output)
                        expmap_pb = angle_trans(poses_batch)

                    # expmap_gt = np.reshape(expmap_gt, (pred_len, 33, 3))
                    # expmap_hmp = np.reshape(expmap_hmp, (pred_len, 33, 3))
                    # expmap_mg = np.squeeze(np.transpose(expmap_mg, (0, 2, 1, 3)), axis=0)
                    # expmap_pb = np.squeeze(np.transpose(expmap_pb, (0, 2, 1, 3)), axis=0)
                    #
                    # eul_gt = em2eul(expmap_gt)
                    # eul_hmp = em2eul(expmap_hmp)
                    # eul_mg = em2eul(expmap_mg)
                    # eul_pb = em2eul(expmap_pb)
                    #
                    # eul_gt = np.reshape(eul_gt, (pred_len, 99))
                    # eul_hmp = np.reshape(eul_hmp, (pred_len, 99))
                    # eul_mg = np.reshape(eul_mg, (pred_len * 2, int(eul_mg.shape[1]) * 3))
                    # eul_pb = np.reshape(eul_pb, (pred_len * 2, int(eul_pb.shape[1]) * 3))
                    #
                    # eul_hmp[:, 0:6] = 0
                    # idx_to_use = np.where(np.std(eul_hmp, 0) > 1e-4)[0]
                    #
                    # eul_gt = eul_gt[:, idx_to_use]
                    # eul_hmp = eul_hmp[:, idx_to_use]
                    # if 'expmaps' in configs[0].data_set:
                    #     eul_mg = eul_mg[:, idx_to_use]
                    #     eul_pb = eul_pb[:, idx_to_use]

                        # gt_diff = np.sum(np.abs(eul_gt - eul_pb[pred_len:, :]))
                        # if gt_diff > 1e-4:
                        #     print("WARNING: gt differs more than it should : ", gt_diff)

                    # mean_errors_hmp[i, :] = euc_error(eul_gt, eul_hmp)
                    # mean_errors_mg[i, :] = euc_error(eul_pb[pred_len:, :], eul_mg[pred_len:, :])

                    coords_gt = flat_edm(to_coords(expmap_gt))
                    coords_hmp = flat_edm(to_coords(expmap_hmp))
                    coords_pb = flat_edm(poses_batch[:, :, pred_len:, :])
                    coords_mg = flat_edm(gen_output[:, :, pred_len:, :])

                    mean_errors_hmp[i, :] = euc_error(coords_gt, coords_hmp)
                    mean_errors_mg[i, :] = euc_error(coords_pb, coords_mg)

                # rec_mean_mean_error = np.array(sample_file['mean_{0}_error'.format(action)], dtype=np.float32)
                # rec_mean_mean_error = rec_mean_mean_error[range(4, np.int(rec_mean_mean_error.shape[0]), 5)]
                mean_mean_errors_hmp = np.mean(mean_errors_hmp, 0)
                mean_mean_errors_mg = np.mean(mean_errors_mg, 0)

                print(action)
                # err_strs = [(Fore.BLUE if np.mean(np.abs(err1 - err2)) < 1e-4 else Fore.YELLOW) + str(np.mean(err1)) + ', ' + str(np.mean(err2))
                #             for err1, err2 in zip(rec_mean_mean_error, mean_mean_errors_hmp)]

                err_strs = [(Fore.GREEN if np.mean((err1 > err2).astype('float32')) > 0.5 else Fore.RED) + str(np.mean(err1)) + ', ' + str(np.mean(err2))
                             for err1, err2 in zip(mean_mean_errors_hmp, mean_mean_errors_mg)]

                for err_str in err_strs:
                    print(err_str)

                print(Style.RESET_ALL)

    elif FLAGS.test_mode == "paper_metrics":

        total_samples = 2 ** 14
        test_mode = False

        if FLAGS.mask_mode != 1 or FLAGS.keep_prob != 0.5:
            print("Warning: this test was designed to work with: -mask_mode 1 -keep_prob 0.5")

        seq_tails_train = np.empty((total_samples, njoints, seq_len, 3))
        labs_train = np.empty((total_samples,))
        t = trange(total_samples // batch_size)
        for i in t:
            labs_batch, poses_batch = train_generator.next()

            poses_batch = poses_batch[..., :3]
            labels = np.reshape(labs_batch[:, 2], (batch_size, 1))

            seq_tails_train[i * batch_size:(i+1) * batch_size, ...] = poses_batch[:, :, seq_len // 2:, :]
            labs_train[i * batch_size:(i+1) * batch_size] = labels[:, 0]

        seq_tails_val = np.empty((total_samples, njoints, seq_len, 3))
        gen_tails_val = [np.empty((total_samples, njoints, seq_len, 3)) for _ in range(len(model_wraps))]
        labs_val = np.empty((total_samples,))
        t = trange(total_samples // batch_size)
        for i in t:
            labs_batch, poses_batch = val_generator.next()

            # mask_batch = poses_batch[..., 3, np.newaxis]
            mask_batch = gen_mask(FLAGS.mask_mode, FLAGS.keep_prob,batch_size, njoints, seq_len, body_members, test_mode)
            poses_batch = poses_batch[..., :3]
            labels = np.reshape(labs_batch[:, 2], (batch_size, 1))

            for m, model_wrap in enumerate(model_wraps):
                gen_output = np.zeros((batch_size, njoints, seq_len, 3))
                for p in range(2):
                    if p == 0:
                        poses_batch_chunk = poses_batch[:, :, :seq_len, :].copy()
                    else:
                        poses_batch_chunk = gen_output.copy()
                    gen_inputs = [poses_batch_chunk, mask_batch]
                    if configs[m].action_cond:
                        gen_inputs.append(labels)
                    if configs[m].latent_cond_dim > 0:
                        latent_noise = gen_latent_noise(batch_size, configs[m].latent_cond_dim)
                        gen_inputs.append(latent_noise)
                    gen_output[:, :, p * (seq_len // 2):(p+1) * (seq_len // 2), :] = \
                        model_wrap.gen_model.predict(gen_inputs, batch_size)[:, :, seq_len // 2:, :]
                gen_tails_val[m][i * batch_size:(i+1) * batch_size, ...] = gen_output

            seq_tails_val[i * batch_size:(i+1) * batch_size, ...] = poses_batch[:, :, seq_len // 2:, :]
            labs_val[i * batch_size:(i+1) * batch_size] = labels[:, 0]

        if configs[0].normalize_data:
            seq_tails_train = data_input.unnormalize_poses(seq_tails_train)
            seq_tails_val = data_input.unnormalize_poses(seq_tails_val)
            for m in range(len(model_wraps)):
                gen_tails_val[m] = data_input.unnormalize_poses(gen_tails_val[m])

        def compute_ent_metrics(gt_seqs, seqs, format='coords'):
            for seq_start, seq_end in [(s * (seq_len // 4), (s+1) * (seq_len // 4)) for s in range(4)] + [(0, seq_len)]:
                gt_seqs_tmp = gt_seqs[:, :, seq_start:seq_end, :]
                seqs_tmp = seqs[:, :, seq_start:seq_end, :]
                if format == 'coords':
                    gt_cent_seqs = gt_seqs_tmp - gt_seqs_tmp[:, 0, np.newaxis, :, :]
                    gt_angle_expmaps = angle_trans(gt_cent_seqs)
                    cent_seqs = seqs_tmp - seqs_tmp[:, 0, np.newaxis, :, :]
                    angle_expmaps = angle_trans(cent_seqs)
                elif format == 'expmaps':
                    gt_angle_expmaps = gt_seqs_tmp
                    angle_expmaps = seqs_tmp

                gt_angle_seqs = npangles.rotmat_to_euler(npangles.expmap_to_rotmat(gt_angle_expmaps))
                angle_seqs = npangles.rotmat_to_euler(npangles.expmap_to_rotmat(angle_expmaps))

                gt_seqs_fft = np.fft.fft(gt_angle_seqs, axis=2)
                gt_seqs_ps = np.abs(gt_seqs_fft) ** 2

                gt_seqs_ps_global = gt_seqs_ps.sum(axis=0) + 1e-8
                gt_seqs_ps_global /= gt_seqs_ps_global.sum(axis=1, keepdims=True)

                seqs_fft = np.fft.fft(angle_seqs, axis=2)
                seqs_ps = np.abs(seqs_fft) ** 2

                seqs_ps_global = seqs_ps.sum(axis=0) + 1e-8
                seqs_ps_global /= seqs_ps_global.sum(axis=1, keepdims=True)

                seqs_ent_global = -np.sum(seqs_ps_global * np.log(seqs_ps_global), axis=1)
                # print("PS Entropy: ", seqs_ent_global.mean())

                seqs_kl_gen_gt = np.sum(seqs_ps_global * np.log(seqs_ps_global / gt_seqs_ps_global), axis=1)
                # print("PS KL(Gen|GT): ", seqs_kl_gen_gt.mean())
                seqs_kl_gt_gen = np.sum(gt_seqs_ps_global * np.log(gt_seqs_ps_global / seqs_ps_global), axis=1)
                # print("PS KL(GT|Gen): ", seqs_kl_gt_gen.mean())

                print(#"frames: ", (seq_start, seq_end),
                      "%.5f & %.5f & %.5f" % (seqs_ent_global.mean(), seqs_kl_gen_gt.mean(), seqs_kl_gt_gen.mean()))


        import matplotlib
        matplotlib.use('Agg')
        actions = ['directions', 'discussion', 'eating', 'greeting', 'phoning',
                   'posing', 'purchases', 'sitting', 'sitting down', 'smoking',
                   'taking photo', 'waiting', 'walking', 'walking dog', 'walking together']

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        # for lab in sorted(set(labs_train)):
        #     idxs = labs_train == lab
        #     xs = seq_tails_train_trans[idxs, 0]
        #     ys = seq_tails_train_trans[idxs, 1]
        #     ax.scatter(xs, ys, marker='.', alpha=0.1, label=str(lab))
        # ax.legend()
        # ax.set_title('Projected samples (GT train classes)')
        # ax.grid(True)
        # fig.tight_layout()
        # plt.show(block=False)

        # fig, ax = plt.subplots()
        # for lab in sorted(set(labs_val)):
        #     idxs = labs_val == lab
        #     xs = seq_tails_val_trans[idxs, 0]
        #     ys = seq_tails_val_trans[idxs, 1]
        #     ax.scatter(xs, ys, marker='.', alpha=0.1, label=actions[int(lab)]) #str(lab))
        #     ax.set_xlim([-4, 10])
        #     ax.set_ylim([-5, 5])
        # ax.legend()
        # ax.set_title('Projected Samples (GT Val Classes)')
        # ax.grid(True)
        # fig.tight_layout()
        # # plt.show(block=False)
        # fig.savefig(images_path+"val_plot.png", dpi=80)

        # fig, ax = plt.subplots()
        # ax.scatter(seq_tails_train_trans[:, 0], seq_tails_train_trans[:, 1], marker='.', alpha=0.1, label='train')
        # ax.legend()
        # ax.scatter(seq_tails_val_trans[:, 0], seq_tails_val_trans[:, 1], marker='.', alpha=0.1, label='val')
        # ax.legend()
        # ax.set_title('Projected samples (GT Splits)')
        # ax.grid(True)
        # fig.tight_layout()
        # plt.show(block=False)

        # Checking sanity of metric
        print("training set sanity check")
        print("PS Entropy, PS KL(Gen|GT), PS KL(GT|Gen)")
        compute_ent_metrics(seq_tails_train, seq_tails_train)

        print("validation vs training set")
        compute_ent_metrics(seq_tails_train, seq_tails_val)

        ### Compute comparison with HMP dataset, only valid for h36 models
        if configs[0].data_set == "Human36":
            print("HMP Baseline")
            from utils.human36_expmaps_to_h5 import actions

            def subsample(seq):
                return seq[range(0, int(seq.shape[0]), 5), :]

            def prepare_expmap(expmap):
                expmap = subsample(expmap)
                expmap = expmap.reshape((expmap.shape[0], 33, 3))
                ex_std = expmap.std(0)
                dim_to_use = np.where((ex_std >= 1e-4).all(axis=-1))[0]
                expmap = expmap[:, dim_to_use, :]
                expmap = expmap.transpose((1, 0, 2))
                return expmap


            expmaps_hmp_gt = []
            expmaps_hmp = []
            with h5.File('../human-motion-prediction/samples.h5', "r") as sample_file:
                for act_idx, action in enumerate(actions):
                    for i in np.arange(8):
                        expmap_hmp_gt = np.array(sample_file['expmap/gt/{1}_{0}'.format(i, action)], dtype=np.float32)
                        expmaps_hmp_gt.append(prepare_expmap(expmap_hmp_gt))
                        expmap_hmp = np.array(sample_file['expmap/preds/{1}_{0}'.format(i, action)], dtype=np.float32)
                        expmaps_hmp.append(prepare_expmap(expmap_hmp))

            compute_ent_metrics(np.stack(expmaps_hmp_gt, 0), np.stack(expmaps_hmp, 0), 'expmaps')

        # teaser_0 = lda_transform(data_input.normalize_poses(np.load("save/motiongan_v7_action_nogan_fp_h36_test_images_gif/survey_3_026.npy"))[:, :, seq_len // 2:, :])
        # teaser_1 = lda_transform(data_input.normalize_poses(np.load("save/motiongan_v7_action_nogan_fp_h36_test_images_gif/survey_3_029.npy"))[:, :, seq_len // 2:, :])

        for m, _ in enumerate(model_wraps):
            print(configs[m].save_path)

            # fig, ax = plt.subplots()
            # ax.scatter(seq_tails_val_trans[:, 0], seq_tails_val_trans[:, 1], marker='.', alpha=0.1, label='GT')
            # ax.legend()
            # ax.scatter(gen_trans[:, 0], gen_trans[:, 1], marker='.', alpha=0.1, label="STMI-GAN")#configs[m].save_path)
            # ax.legend()

            # ax.scatter(seq_tails_val_trans[2001, np.newaxis, 0], seq_tails_val_trans[2001, np.newaxis, 1], marker='x', alpha=1.0, label='GT seq#2001')
            # ax.legend()
            # ax.scatter(gen_trans[2001, np.newaxis, 0], gen_trans[2001, np.newaxis, 1], marker='x', alpha=1.0, label="STMI-GAN seq#2001")#configs[m].save_path)
            # ax.legend()
            #
            # print("seq#2001 pred dist:", np.sqrt(np.sum(np.square(gen_trans[2001, :] - seq_tails_val_trans[2001, :]))))

            # ax.scatter(teaser_0[0, np.newaxis, 0], teaser_0[0, np.newaxis, 1], marker='x', alpha=1.0, label='teaser GT')
            # ax.legend()
            # ax.scatter(teaser_0[1, np.newaxis, 0], teaser_0[1, np.newaxis, 1], marker='x', alpha=1.0, label='teaser Gen')
            # ax.legend()

            # print("teaser0 pred dist:", np.sqrt(np.sum(np.square(teaser_0[0, :] - teaser_0[1, :]))))

            # ax.scatter(teaser_1[0, np.newaxis, 0], teaser_1[0, np.newaxis, 1], marker='x', alpha=1.0, label='teaser 1 GT')
            # ax.legend()
            # ax.scatter(teaser_1[1, np.newaxis, 0], teaser_1[1, np.newaxis, 1], marker='x', alpha=1.0, label='teaser 1 Gen')
            # ax.legend()

            # print("teaser1 pred dist:", np.sqrt(np.sum(np.square(teaser_1[0, :] - teaser_1[1, :]))))

            # ax.set_title('Projected Samples')
            # ax.grid(True)
            # ax.set_xlim([-4, 10])
            # ax.set_ylim([-5, 5])

            # fig.tight_layout()
            # plt.show(block=False)
            # fig.savefig(images_path + ("gen_plot_%d.png" % m), dpi=320)

            compute_ent_metrics(seq_tails_val, gen_tails_val[m])

        # plt.show()

        # Per Action Metrics
        # actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
        #            'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking',
        #            'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
        #
        # for lab in sorted(set(labs_train)):
        #     print('\nAction: ' + actions[int(lab)])
        #     idxs = labs_val == lab
        #     seq_tails_val_act = seq_tails_val[idxs, ...]
        #
        #     for m, _ in enumerate(model_wraps):
        #         print(configs[m].save_path)
        #
        #         gen_tails_val_act = gen_tails_val[m][idxs, ...]
        #
        #         compute_ent_metrics(seq_tails_val_act, gen_tails_val_act)

    elif FLAGS.test_mode == "plot_survey":

        print('models loaded in the following order:')
        for config in configs:
            print(config.save_path)

        print('expecting nogan baseline as 0 and complex model as 1')

        from utils.human36_expmaps_to_h5 import actions
        h36_coords_used_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
        parent, offset, rotInd, expmapInd = _some_variables()

        def subsample(seq):
            return seq[range(0, int(seq.shape[0]), 5), :]

        def to_coords(seq_angles):
            seq_coords = np.empty((1, len(h36_coords_used_joints), seq_angles.shape[0], 3))
            for i in range(seq_angles.shape[0]):
                frame_coords = fkl(seq_angles[i, :], parent, offset, rotInd, expmapInd)
                seq_coords[0, :, i, :] = frame_coords[h36_coords_used_joints, :]
            seq_coords[..., 1] = seq_coords[..., 1] * -1  # Inverting y axis for visualization purposes
            return seq_coords

        def gen_batch(mask_mode, keep_prob):
            FLAGS.mask_mode = mask_mode
            FLAGS.keep_prob = keep_prob

            labs_batch, poses_batch, mask_batch = get_inputs(False)

            gen_outputs = []
            for m, model_wrap in enumerate(model_wraps):
                gen_inputs = [poses_batch, mask_batch]
                if configs[m].action_cond:
                    labels = np.reshape(labs_batch[:, 2], (batch_size, 1))
                    gen_inputs.append(labels)
                if configs[m].latent_cond_dim > 0:
                    latent_noise = gen_latent_noise(batch_size, configs[m].latent_cond_dim)
                    gen_inputs.append(latent_noise)
                gen_output = model_wrap.gen_model.predict(gen_inputs, batch_size)
                if configs[m].normalize_data:
                    gen_output = data_input.unnormalize_poses(gen_output)
                gen_outputs.append(gen_output)

            if configs[0].normalize_data:
                poses_batch = data_input.unnormalize_poses(poses_batch)

            return poses_batch, mask_batch, gen_outputs

        print("plotting survey 0")
        # Generating Martinez etal baseline

        order = np.random.binomial(1, 0.5, size=batch_size)
        np.savetxt(images_path + "survey_0_order.csv", order, delimiter=",")

        with h5.File('../human-motion-prediction/samples.h5', "r") as sample_file:
            for act_idx, action in enumerate(actions):
                pred_len = seq_len // 2
                mean_errors_hmp = np.zeros((8, pred_len))
                mean_errors_mg = np.zeros((8, pred_len))
                for i in np.arange(8):
                    seq_idx = (act_idx * 8) + i

                    encoder_inputs = np.array(sample_file['expmap/encoder_inputs/{1}_{0}'.format(i, action)], dtype=np.float32)
                    decoder_inputs = np.array(sample_file['expmap/decoder_inputs/{1}_{0}'.format(i, action)], dtype=np.float32)
                    # decoder_outputs = np.array(sample_file['expmap/decoder_outputs/{1}_{0}'.format(i, action)], dtype=np.float32)
                    input_seeds_sact = np.int32(sample_file['expmap/input_seeds_sact/{1}_{0}'.format(i, action)])
                    input_seeds_idx = np.int32(sample_file['expmap/input_seeds_idx/{1}_{0}'.format(i, action)])
                    input_seeds_seqlen = np.int32(sample_file['expmap/input_seeds_seqlen/{1}_{0}'.format(i, action)])

                    seq_angles = np.concatenate([encoder_inputs, decoder_inputs[np.newaxis, 0, :]], axis=0)
                    # seq_angles = np.concatenate([encoder_inputs, decoder_inputs[np.newaxis, 0, :], decoder_outputs], axis=0)
                    # seq_angles = subsample(seq_angles)
                    # seq_angles = seq_angles[10 - pred_len:10 + pred_len, :]

                    expmap_gt = np.array(sample_file['expmap/gt/{1}_{0}'.format(i, action)], dtype=np.float32)
                    expmap_gt = np.concatenate([seq_angles, expmap_gt], axis=0)
                    expmap_gt = subsample(expmap_gt)
                    expmap_gt = expmap_gt[10 - pred_len:10 + pred_len, :]
                    coords_gt = to_coords(expmap_gt)

                    expmap_hmp = np.array(sample_file['expmap/preds/{1}_{0}'.format(i, action)], dtype=np.float32)
                    expmap_hmp = np.concatenate([seq_angles, expmap_hmp], axis=0)
                    expmap_hmp = subsample(expmap_hmp)
                    expmap_hmp = expmap_hmp[10 - pred_len:10 + pred_len, :]
                    coords_gen = to_coords(expmap_hmp)

                    # labs_batch = np.array([input_seeds_idx, 6, act_idx, input_seeds_seqlen])

                    if order[seq_idx] == 0:
                        coords = np.concatenate([coords_gt, coords_gen])
                    else:
                        coords = np.concatenate([coords_gen, coords_gt])
                    save_path = images_path + ("survey_0/gt_%d/%03d.gif" % (order[seq_idx], seq_idx))
                    plot_seq_gif(coords, None, configs[0].data_set, save_path=save_path, figwidth=512, figheight=256)

        print("plotting survey 1")
        order = np.random.binomial(1, 0.5, size=batch_size)
        np.savetxt(images_path + "survey_1_order.csv", order, delimiter=",")

        coords_gt, _, gen_outputs = gen_batch(1, 0.5)

        coords_gt = coords_gt - coords_gt[:, 0, np.newaxis, :, :]
        coords_gen = gen_outputs[1] - gen_outputs[1][:, 0, np.newaxis, :, :]

        for seq_idx in range(len(actions) * 8):
            save_path = images_path + ("survey_1/gt_%d/%03d.gif" % (order[seq_idx], seq_idx))
            if order[seq_idx] == 0:
                coords = np.concatenate([coords_gt[np.newaxis, seq_idx, ...], coords_gen[np.newaxis, seq_idx, ...]])
            else:
                coords = np.concatenate([coords_gen[np.newaxis, seq_idx, ...], coords_gt[np.newaxis, seq_idx, ...]])
            plot_seq_gif(coords, None, configs[0].data_set, save_path=save_path, figwidth=512, figheight=256)


        print("plotting survey 2 and 3")
        order = np.random.binomial(1, 0.5, size=batch_size)
        np.savetxt(images_path + "survey_2_order.csv", order, delimiter=",")
        np.savetxt(images_path + "survey_3_order.csv", order, delimiter=",")

        coords_gt, _, gen_outputs = gen_batch(1, 0.5)

        coords_gen_0 = gen_outputs[0]
        coords_gen_1 = gen_outputs[1]

        for seq_idx in range(len(actions) * 8):
            save_path_0 = images_path + ("survey_2/gt_%d/%03d.gif" % (order[seq_idx], seq_idx))
            save_path_1 = images_path + ("survey_3/gt_%d/%03d.gif" % (order[seq_idx], seq_idx))
            # save_path_1 = images_path + ("survey_1/gt_%d/%03d.png" % (order[seq_idx], seq_idx))

            if order[seq_idx] == 0:
                coords_0 = np.concatenate([coords_gt[np.newaxis, seq_idx, ...], coords_gen_0[np.newaxis, seq_idx, ...]])
                coords_1 = np.concatenate([coords_gt[np.newaxis, seq_idx, ...], coords_gen_1[np.newaxis, seq_idx, ...]])
            else:
                coords_0 = np.concatenate([coords_gen_0[np.newaxis, seq_idx, ...], coords_gt[np.newaxis, seq_idx, ...]])
                coords_1 = np.concatenate([coords_gen_1[np.newaxis, seq_idx, ...], coords_gt[np.newaxis, seq_idx, ...]])

            plot_seq_gif(coords_0, None, configs[0].data_set, save_path=save_path_0, figwidth=512, figheight=256)
            plot_seq_gif(coords_1, None, configs[0].data_set, save_path=save_path_1, figwidth=512, figheight=256)
            # plot_seq_frozen(coords_1, None, configs[0].data_set, save_path=save_path_1, figwidth=512, figheight=256)
            np.save(images_path + ("survey_3_%03d.npy" % seq_idx), coords_1)

    elif FLAGS.test_mode == "alternate_seq_dist":

        n_futures = 32
        total_samples = 2 ** 10

        PROBS = np.arange(0.0, 1.1, 0.2)

        dist_table = np.zeros((len(PROBS), total_samples // batch_size, len(model_wraps)))
        for p, prob in enumerate(PROBS):
            FLAGS.mask_mode = 1
            FLAGS.keep_prob = prob

            for b in trange(total_samples // batch_size):
                labs_batch, poses_batch, mask_batch = get_inputs(False)

                for m, model_wrap in enumerate(model_wraps):
                    gen_outputs = []
                    l2diffs = []
                    for f in range(n_futures):
                        gen_inputs = [poses_batch, mask_batch]
                        if configs[m].action_cond:
                            labels = np.reshape(labs_batch[:, 2], (batch_size, 1))
                            gen_inputs.append(labels)
                        if configs[m].latent_cond_dim > 0:
                            latent_noise = gen_latent_noise(batch_size, configs[m].latent_cond_dim)
                            gen_inputs.append(latent_noise)
                        gen_output = model_wrap.gen_model.predict(gen_inputs, batch_size)
                        if configs[m].normalize_data:
                            gen_output = data_input.unnormalize_poses(gen_output)
                        gen_outputs.append(gen_output.reshape((batch_size * njoints * seq_len, 3)))
                        if f > 0:
                            for g in range(f):
                                l2diff = np.mean(np.sqrt(np.sum((gen_outputs[g] - gen_outputs[f]) ** 2, -1)))
                                l2diffs.append(l2diff)

                    dist_table[p, b, m] = np.mean(l2diffs)

        print(dist_table.mean(1))

    elif FLAGS.test_mode == "alternate_seq_im":

        n_futures = 8

        for i in trange(val_batches):
            labs_batch, poses_batch, mask_batch = get_inputs(False)

            gen_outputs = []
            for m, model_wrap in enumerate(model_wraps):
                for f in range(n_futures):
                    gen_inputs = [poses_batch, mask_batch]
                    if configs[m].action_cond:
                        labels = np.reshape(labs_batch[:, 2], (batch_size, 1))
                        gen_inputs.append(labels)
                    if configs[m].latent_cond_dim > 0:
                        latent_noise = gen_latent_noise(batch_size, configs[m].latent_cond_dim)
                        gen_inputs.append(latent_noise)
                    gen_output = model_wrap.gen_model.predict(gen_inputs, batch_size)
                    if configs[m].normalize_data:
                        gen_output = data_input.unnormalize_poses(gen_output)
                    gen_outputs.append(gen_output)

            if configs[0].normalize_data:
                poses_batch = data_input.unnormalize_poses(poses_batch)

            for j in range(batch_size):
                seq_idx = j

                if FLAGS.images_mode == "gif":
                    plot_func = plot_seq_gif
                    figwidth = 512 * (len(configs) + 1)
                    figheight = 512
                elif FLAGS.images_mode == "png":
                    plot_func = plot_seq_frozen  # plot_seq_pano
                    figwidth = 768
                    figheight = 256 * (len(configs) + 1)

                save_path = images_path + ("%d_%d.%s" % (i, j, FLAGS.images_mode))
                plot_func(np.concatenate([poses_batch[np.newaxis, seq_idx, ...]] +
                                         [gen_output[np.newaxis, seq_idx, ...] for gen_output in gen_outputs] ),
                          labs_batch[seq_idx, ...],
                          configs[0].data_set,
                          seq_masks=mask_batch[seq_idx, ...],
                          extra_text='mask mode: %s keep prob: %s' % (MASK_MODES[FLAGS.mask_mode], FLAGS.keep_prob),
                          save_path=save_path, figwidth=figwidth, figheight=figheight)

    elif FLAGS.test_mode == "rebuttal":

        def euc_error(x, y):
            x = angle_trans(x)
            y = angle_trans(y)
            return np.sqrt(np.sum(np.square(x - y), 3))

        from utils.human36_expmaps_to_h5 import actions
        h36_coords_used_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
        parent, offset, rotInd, expmapInd = _some_variables()

        def to_coords(seq_angles):
            seq_coords = np.empty((len(h36_coords_used_joints), seq_angles.shape[0], 3))
            for i in range(seq_angles.shape[0]):
                frame_coords = fkl(seq_angles[i, :], parent, offset, rotInd, expmapInd)
                seq_coords[:, i, :] = frame_coords[h36_coords_used_joints, :]
            seq_coords[..., 1] = seq_coords[..., 1] * -1  # Inverting y axis for visualization purposes
            return seq_coords

        import os

        def rotate_seq(seq):
            for l in range(seq.shape[2]):
                seq[:, :, np.newaxis, l, :], _ = rotate_start(seq[:, :, np.newaxis, l, :], body_members)
            return seq

        def prep_seq(seq):
            frames_idxs = [0, 9, 19]
            seq = seq[:, :, frames_idxs, :]
            seq = rotate_seq(seq)
            seq -= seq[:, np.newaxis, 0, :, :]
            seq[:, :, 0, :2] -= 500
            seq[:, :, -1, :2] += 500
            return seq

        file_path = os.path.join(configs[0].data_path, configs[0].data_set + configs[0].data_set_version + '.h5')
        coords_file = h5.File(file_path, 'r')
        val_keys = [configs[0].data_set + '/Validate/' + k
                    for k in coords_file.get(configs[0].data_set + '/Validate').keys()]

        dist_table = []
        with h5.File('../human-motion-prediction/samples.h5', "r") as expmaps_file:
            for act_idx, action in enumerate(actions):
                pred_len = seq_len // 2
                mean_errors_hmp = np.zeros((8, pred_len))
                mean_errors_mg = np.zeros((8, pred_len))
                for i in np.arange(8):
                    seq_idx = (act_idx * 8) + i

                    encoder_inputs = np.array(expmaps_file['expmap/encoder_inputs/{1}_{0}'.format(i, action)], dtype=np.float32)
                    decoder_inputs = np.array(expmaps_file['expmap/decoder_inputs/{1}_{0}'.format(i, action)], dtype=np.float32)
                    # decoder_outputs = np.array(sample_file['expmap/decoder_outputs/{1}_{0}'.format(i, action)], dtype=np.float32)
                    input_seeds_sact = np.int32(expmaps_file['expmap/input_seeds_sact/{1}_{0}'.format(i, action)])
                    input_seeds_idx = np.int32(expmaps_file['expmap/input_seeds_idx/{1}_{0}'.format(i, action)])
                    input_seeds_seqlen = np.int32(expmaps_file['expmap/input_seeds_seqlen/{1}_{0}'.format(i, action)])

                    seq_angles = np.concatenate([encoder_inputs, decoder_inputs[np.newaxis, 0, :]], axis=0)
                    # seq_angles = np.concatenate([encoder_inputs, decoder_inputs[np.newaxis, 0, :], decoder_outputs], axis=0)
                    # seq_angles = subsample(seq_angles)
                    # seq_angles = seq_angles[10 - pred_len:10 + pred_len, :]

                    expmap_gt = np.array(expmaps_file['expmap/gt/{1}_{0}'.format(i, action)], dtype=np.float32)
                    expmap_gt = np.concatenate([seq_angles, expmap_gt], axis=0)
                    expmap_gt = expmap_gt[range(0, int(expmap_gt.shape[0]), 5), :]
                    expmap_gt = expmap_gt[10 - pred_len:10 + pred_len, :]
                    coords_expmap_gt = to_coords(expmap_gt)
                    coords_expmap_gt = coords_expmap_gt[np.newaxis, ...]

                    expmap_hmp = np.array(expmaps_file['expmap/preds/{1}_{0}'.format(i, action)], dtype=np.float32)
                    expmap_hmp = np.concatenate([seq_angles, expmap_hmp], axis=0)
                    expmap_hmp = expmap_hmp[range(0, int(expmap_hmp.shape[0]), 5), :]
                    expmap_hmp = expmap_hmp[10 - pred_len:10 + pred_len, :]
                    coords_expmap_hmp = to_coords(expmap_hmp)
                    coords_expmap_hmp = coords_expmap_hmp[np.newaxis, ...]

                    for key in val_keys:
                        coords_act_idx = np.int32(coords_file[key + '/Action']) - 1  # Small hack to reindex the classes from 0
                        if coords_act_idx == act_idx:
                            coords_sact = np.int32(coords_file[key + '/Subaction'])
                            coords_subject = np.int32(coords_file[key + '/Subject'])
                            coords_pose = np.array(coords_file[key + '/Pose'], dtype=np.float32)
                            coords_pose = coords_pose[..., range(0, coords_pose.shape[2], 2)]
                            coords_seq_len = np.int32(coords_pose.shape[2])

                            if input_seeds_sact == coords_sact and coords_seq_len == input_seeds_seqlen:
                                coords_pose = coords_pose[..., input_seeds_idx:input_seeds_idx+100]
                                coords_pose = coords_pose[..., range(0, int(coords_pose.shape[2]), 5)]
                                coords_pose = np.transpose(coords_pose, (0, 2, 1))
                                coords_pose = coords_pose[h36_coords_used_joints, ...]
                                coords_pose = coords_pose[np.newaxis, ...]

                                poses_batch = coords_pose
                                mask_batch = gen_mask(1, 0.5, batch_size, njoints, seq_len, body_members, False)

                                gen_outputs = []
                                for m, model_wrap in enumerate(model_wraps):
                                    if configs[m].normalize_data:
                                        poses_batch = data_input.normalize_poses(poses_batch)
                                    gen_inputs = [poses_batch, mask_batch]
                                    if configs[m].action_cond:
                                        labels = np.reshape(act_idx, (batch_size, 1))
                                        gen_inputs.append(labels)
                                    if configs[m].latent_cond_dim > 0:
                                        latent_noise = gen_latent_noise(batch_size, configs[m].latent_cond_dim)
                                        gen_inputs.append(latent_noise)
                                    gen_output = model_wrap.gen_model.predict(gen_inputs, batch_size)
                                    if configs[m].normalize_data:
                                        gen_output = data_input.unnormalize_poses(gen_output)
                                    gen_outputs.append(gen_output)

                                # To add global position
                                # coords_expmap_gt += coords_pose[:, np.newaxis, 0, :, :]
                                # coords_expmap_hmp += coords_pose[:, np.newaxis, 0, :, :]

                                save_path = images_path + ("rebuttal/%03d.png" % seq_idx)
                                coords = np.concatenate([#coords_expmap_gt[:,:,frames_idxs,:],
                                                         prep_seq(coords_pose),
                                                         prep_seq(coords_expmap_hmp),] +
                                                         [prep_seq(gen_output) for gen_output in gen_outputs], axis=0)
                                plot_seq_frozen(coords, None, configs[0].data_set, save_path=save_path, figwidth=256 * coords.shape[0], figheight=256)
                                # plot_seq_pano(coords, None, configs[0].data_set, save_path=save_path, figwidth=256 * coords.shape[0], figheight=256)
                                dist_table.append(np.stack([seq_idx] +
                                                           [np.mean(euc_error(coords_expmap_gt, coords_expmap_hmp))] +
                                                           [np.mean(euc_error(coords_pose, gen_output)) for gen_output in gen_outputs]))
        np.savetxt(images_path + 'rebuttal/dists.txt', np.stack(dist_table), '%.4f', ',', '\n')


from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import scipy as sp
from config import get_config
from data_input import DataInput
from models.motiongan import get_model
from models.dmnn import DMNNv1
from utils.restore_keras_model import restore_keras_model
from utils.viz import plot_seq_gif, plot_seq_pano
from utils.seq_utils import MASK_MODES, gen_mask, linear_baseline, burke_baseline, post_process, seq_to_angles_transformer, get_angles_mask, gen_latent_noise
import h5py as h5
from tqdm import trange
from collections import OrderedDict
from colorama import Fore, Back, Style
import utils.npangles as npangles

logging = tf.logging
flags = tf.flags
flags.DEFINE_bool("verbose", False, "To talk or not to talk")
flags.DEFINE_multi_string("model_path", None, "Model output directory")
flags.DEFINE_string("test_mode", "show_images", "Test modes: show_images, write_images, write_data, dmnn_score, dmnn_score_table, hmp_compare, dist_compare")
flags.DEFINE_string("dmnn_path", None, "Path to trained DMNN model")
flags.DEFINE_string("images_mode", "gif", "Image modes: gif, png")
flags.DEFINE_integer("mask_mode", 3, "Mask modes: " + ' '.join(['%d:%s' % tup for tup in enumerate(MASK_MODES)]))
flags.DEFINE_float("keep_prob", 0.8, "Probability of keeping input data. (1 == Keep All)")
FLAGS = flags.FLAGS


def _reset_rand_seed():
    seed = 42
    np.random.seed(seed)
    tf.set_random_seed(seed)


if __name__ == "__main__":
    _reset_rand_seed()
    # Config stuff
    batch_size = 1 if ((not "dmnn_score" in FLAGS.test_mode) and
                       (not "dist_compare" in FLAGS.test_mode)) else 256
    configs = []
    model_wraps = []
    # Hacks to fill undefined, but necessary flags
    tf.flags.DEFINE_string("config_file", None, None)
    tf.flags.DEFINE_string("save_path", None, None)

    for save_path in FLAGS.model_path:
        FLAGS.save_path = save_path
        config = get_config(FLAGS)
        config.only_val = True if not "dist_compare" in FLAGS.test_mode else False
        config.batch_size = batch_size

        # Model building
        if config.model_type == 'motiongan':
            model_wrap = get_model(config)

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
    data_input = DataInput(configs[0])
    _reset_rand_seed()
    train_batches = data_input.train_epoch_size
    train_generator = data_input.batch_generator(True)
    val_batches = data_input.val_epoch_size
    val_generator = data_input.batch_generator(False)

    if FLAGS.test_mode == "write_images":
        images_path = "%s_test_images_%s/" % \
                      ('_'.join(FLAGS.model_path), FLAGS.images_mode)
        if not tf.gfile.Exists(images_path):
            tf.gfile.MkDir(images_path)

    njoints = configs[0].njoints
    seq_len = model_wraps[0].seq_len
    body_members = configs[0].body_members  # if not configs[0].data_set == 'Human36' else configs[0].full_body_members
    angle_trans = seq_to_angles_transformer(body_members)

    def get_inputs():
        labs_batch, poses_batch = val_generator.next()

        mask_batch = poses_batch[..., 3, np.newaxis]
        mask_batch = mask_batch * gen_mask(FLAGS.mask_mode, FLAGS.keep_prob,
                                           batch_size, njoints, seq_len, body_members, True)
        poses_batch = poses_batch[..., :3]

        gen_inputs = [poses_batch, mask_batch]

        return labs_batch, poses_batch, mask_batch, gen_inputs

    if "images" in FLAGS.test_mode:

        for i in trange(val_batches):
            labs_batch, poses_batch, mask_batch, gen_inputs = get_inputs()
            labels = np.reshape(labs_batch[:, 2], (batch_size, 1))

            gen_outputs = []
            proc_gen_outputs = []
            for m, model_wrap in enumerate(model_wraps):
                if configs[m].action_cond:
                    gen_inputs.append(labels)
                gen_output = model_wrap.gen_model.predict(gen_inputs, batch_size)
                if configs[m].action_cond:
                    gen_inputs.pop(-1)
                proc_gen_output = np.empty_like(gen_output)
                for j in range(batch_size):
                    proc_gen_output[j, ...] = post_process(poses_batch[j, ...], gen_output[j, ...],
                                                      mask_batch[j, ...], body_members)
                if configs[m].normalize_data:
                    gen_output = data_input.unnormalize_poses(gen_output)
                    proc_gen_output = data_input.unnormalize_poses(proc_gen_output)
                gen_outputs.append(gen_output)
                proc_gen_outputs.append(proc_gen_output)

            if configs[0].normalize_data:
                poses_batch = data_input.unnormalize_poses(poses_batch)

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
                    figheight = 384 * (len(configs) + 1)
                elif FLAGS.images_mode == "png":
                    plot_func = plot_seq_pano
                    figwidth = 768
                    figheight = 384 * 3

                linear_seq =\
                    linear_baseline(poses_batch[seq_idx, ...], mask_batch[seq_idx, ...])
                linear_seq = np.expand_dims(linear_seq, 0)
                burke_seq = \
                    burke_baseline(poses_batch[seq_idx, ...], mask_batch[seq_idx, ...])
                burke_seq = np.expand_dims(burke_seq, 0)

                plot_func(np.concatenate([poses_batch[np.newaxis, seq_idx, ...], linear_seq, burke_seq] +
                                         [gen_output[np.newaxis, seq_idx, ...] for gen_output in gen_outputs] +
                                         [proc_gen_output[np.newaxis, seq_idx, ...] for proc_gen_output in proc_gen_outputs]),
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
            labels = np.reshape(labs_batch[:, 2], (batch_size, 1))

            for m, model_wrap in enumerate(model_wraps):
                if configs[m].action_cond:
                    gen_inputs.append(labels)
                gen_output = model_wrap.gen_model.predict(gen_inputs, batch_size)
                if configs[m].action_cond:
                    gen_inputs.pop(-1)
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

                labs_batch, poses_batch, mask_batch, gen_inputs = get_inputs()
                labels = np.reshape(labs_batch[:, 2], (batch_size, 1))

                unorm_poses_batch = unnormalize_batch(poses_batch)
                unorm_poses_batch_edm = edm(unorm_poses_batch)
                unorm_poses_batch_angles = angle_trans(unorm_poses_batch)

                p2ps_occ_num = np.sum(1.0 - mask_batch) + 1e-8
                dms_mask_batch = np.expand_dims(mask_batch, axis=1) * np.expand_dims(mask_batch, axis=2)
                dms_occ_num = np.sum(1.0 - dms_mask_batch) + 1e-8
                angles_mask_batch = get_angles_mask(mask_batch, body_members)
                angles_occ_num = np.sum(1.0 - angles_mask_batch) + 1e-8

                for m, model_wrap in enumerate(model_wraps):
                    if configs[m].action_cond:
                        gen_inputs.append(labels)
                    gen_output = model_wrap.gen_model.predict(gen_inputs, batch_size)
                    if configs[m].action_cond:
                        gen_inputs.pop(-1)
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

            PROBS = np.arange(0.0, 1.1, 0.1)

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
    elif FLAGS.test_mode == "hmp_compare":
        from utils.human36_expmaps_to_h5 import actions

        def em2eul(a):
            return npangles.rotmat_to_euler(npangles.expmap_to_rotmat(a))

        def euc_error(x, y):
            return np.sqrt(np.sum(np.square(x - y), 1))

        def motion_error(x, y):
            return euc_error(x[1:, :] - x[:-1, :], y[1:, :] - y[:-1, :])

        def subsample(seq):
            return seq[range(0, int(seq.shape[0]), 5), :]

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
                    # expmap_gt = expmap_gt[4:, ...]  # Our model predicts every 200ms, first frames are not compared
                    # expmap_gt = subsample(expmap_gt)
                    expmap_gt = expmap_gt[:pred_len, ...]

                    expmap_hmp = np.array(sample_file['expmap/preds/{1}_{0}'.format(i, action)], dtype=np.float32)
                    # expmap_hmp = expmap_hmp[4:, ...]
                    # expmap_hmp = subsample(expmap_hmp)
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
                                if plen == input_seeds_seqlen:
                                    pose = pose[:, input_seeds_idx:input_seeds_idx+100, :]
                                    pose = pose[:, range(0, 100, 5), :]
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

                    expmap_gt = np.reshape(expmap_gt, (pred_len, 33, 3))
                    expmap_hmp = np.reshape(expmap_hmp, (pred_len, 33, 3))
                    expmap_mg = np.squeeze(np.transpose(expmap_mg, (0, 2, 1, 3)), axis=0)
                    expmap_pb = np.squeeze(np.transpose(expmap_pb, (0, 2, 1, 3)), axis=0)

                    eul_gt = em2eul(expmap_gt)
                    eul_hmp = em2eul(expmap_hmp)
                    eul_mg = em2eul(expmap_mg)
                    eul_pb = em2eul(expmap_pb)

                    eul_gt = np.reshape(eul_gt, (pred_len, 99))
                    eul_hmp = np.reshape(eul_hmp, (pred_len, 99))
                    eul_mg = np.reshape(eul_mg, (pred_len * 2, int(eul_mg.shape[1]) * 3))
                    eul_pb = np.reshape(eul_pb, (pred_len * 2, int(eul_pb.shape[1]) * 3))

                    eul_hmp[:, 0:6] = 0
                    idx_to_use = np.where(np.std(eul_hmp, 0) > 1e-4)[0]

                    eul_gt = eul_gt[:, idx_to_use]
                    eul_hmp = eul_hmp[:, idx_to_use]
                    if 'expmaps' in configs[0].data_set:
                        eul_mg = eul_mg[:, idx_to_use]
                        eul_pb = eul_pb[:, idx_to_use]

                        # gt_diff = np.sum(np.abs(eul_gt - eul_pb[pred_len:, :]))
                        # if gt_diff > 1e-4:
                        #     print("WARNING: gt differs more than it should : ", gt_diff)

                    mean_errors_hmp[i, :] = euc_error(eul_gt, eul_hmp)
                    mean_errors_mg[i, :] = euc_error(eul_pb[pred_len:, :], eul_mg[pred_len:, :])

                rec_mean_mean_error = np.array(sample_file['mean_{0}_error'.format(action)], dtype=np.float32)
                # rec_mean_mean_error = rec_mean_mean_error[range(4, np.int(rec_mean_mean_error.shape[0]), 5)]
                mean_mean_errors_hmp = np.mean(mean_errors_hmp, 0)
                mean_mean_errors_mg = np.mean(mean_errors_mg, 0)

                print(action)
                err_strs = [(Fore.BLUE if np.mean(np.abs(err1 - err2)) < 1e-4 else Fore.YELLOW) + str(np.mean(err2))
                            for err1, err2 in zip(rec_mean_mean_error, mean_mean_errors_hmp)]

                err_strs += [(Fore.GREEN if np.mean((err1 > err2).astype('float32')) > 0.5 else Fore.RED) + str(np.mean(err2))
                             for err1, err2 in zip(mean_mean_errors_hmp, mean_mean_errors_mg)]

                for err_str in err_strs:
                    print(err_str)

                print(Style.RESET_ALL)


    elif FLAGS.test_mode == "dist_compare":

        total_samples = 2 ** 14

        seq_tails_train = np.empty((total_samples, njoints, (seq_len // 2), 3))
        labs_train = np.empty((total_samples,))
        t = trange(total_samples // batch_size)
        for i in t:
            labs_batch, poses_batch = train_generator.next()

            mask_batch = poses_batch[..., 3, np.newaxis]
            mask_batch = mask_batch * gen_mask(FLAGS.mask_mode, FLAGS.keep_prob,
                                               batch_size, njoints, seq_len,
                                               body_members, True)
            poses_batch = poses_batch[..., :3]

            labels = np.reshape(labs_batch[:, 2], (batch_size, 1))

            seq_tails_train[i * batch_size:(i+1) * batch_size, ...] = poses_batch[:, :, seq_len // 2:, :]
            labs_train[i * batch_size:(i+1) * batch_size] = labels[:, 0]

        # from layers.tsne import *
        # from tensorflow.contrib.keras.api.keras.models import Sequential
        # from tensorflow.contrib.keras.api.keras.layers import Dense
        # from tensorflow.contrib.keras.api.keras.optimizers import SGD
        #
        # X_train = np.reshape(seq_tails_train, (total_samples, -1))
        # d = 2
        #
        # tsne_model = Sequential()
        # tsne_model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
        # tsne_model.add(Dense(512, activation='relu'))
        # tsne_model.add(Dense(512, activation='relu'))
        # tsne_model.add(Dense(d))
        # tsne_model.compile(loss=build_tsne_loss(d, batch_size), optimizer=SGD(lr=0.1))
        #
        # P = compute_joint_probabilities(X_train, batch_size=batch_size, d=d, perplexity=25, tol=1e-5, verbose=0)
        # Y_train = P.reshape(X_train.shape[0], -1)
        #
        # tsne_model.fit(X_train, Y_train, batch_size=batch_size, epochs=1024, shuffle=False, verbose=0)

        seq_tails_val = np.empty((total_samples, njoints, (seq_len // 2), 3))
        gen_tails_val = [np.empty((total_samples, njoints, (seq_len // 2), 3)) for _ in range(len(model_wraps))]
        labs_val = np.empty((total_samples,))
        t = trange(total_samples // batch_size)
        for i in t:
            labs_batch, poses_batch = val_generator.next()

            mask_batch = poses_batch[..., 3, np.newaxis]
            mask_batch = mask_batch * gen_mask(FLAGS.mask_mode, FLAGS.keep_prob,
                                               batch_size, njoints, seq_len,
                                               body_members, True)
            poses_batch = poses_batch[..., :3]

            labels = np.reshape(labs_batch[:, 2], (batch_size, 1))

            for m, model_wrap in enumerate(model_wraps):
                gen_inputs = [poses_batch, mask_batch]
                if configs[m].action_cond:
                    gen_inputs.append(labels)
                gen_output = model_wrap.gen_model.predict(gen_inputs, batch_size)
                gen_tails_val[m][i * batch_size:(i+1) * batch_size, ...] = gen_output[:, :, seq_len // 2:, :]

            seq_tails_val[i * batch_size:(i+1) * batch_size, ...] = poses_batch[:, :, seq_len // 2:, :]
            labs_val[i * batch_size:(i+1) * batch_size] = labels[:, 0]

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lda = LinearDiscriminantAnalysis()
        lda.fit(np.reshape(seq_tails_train, (total_samples, -1)), labs_train)
        print(lda.explained_variance_ratio_)

        def lda_transform(seqs):
            return lda.transform(np.reshape(seqs, (total_samples, -1)))

        seq_tails_train_trans = lda_transform(seq_tails_train)
        seq_tails_val_trans = lda_transform(seq_tails_val)

        from sklearn.neighbors import NearestNeighbors

        def compute_metrics(c1, c2, dist_metric='euclidean'):
            c1 = np.reshape(c1, (c1.shape[0], -1))
            c2 = np.reshape(c2, (c2.shape[0], -1))

            dist_mat_1 = sp.spatial.distance.cdist(c1, c1, metric=dist_metric)
            dist_mat_2 = sp.spatial.distance.cdist(c2, c2, metric=dist_metric)
            dist_mat_12 = sp.spatial.distance.cdist(c1, c2, metric=dist_metric)

            mean_dist_1 = np.mean(dist_mat_1)
            mean_dist_2 = np.mean(dist_mat_2)
            # min_dist_1 = np.mean(np.min(dist_mat_12, axis=0))
            min_dist_2 = np.mean(np.min(dist_mat_12, axis=1))
            # min_dist = (min_dist_1 + min_dist_2) / 2 # symmetric dist
            min_dist = min_dist_2

            pred_dist = np.mean(np.diag(dist_mat_12))

            # knn_1 = NearestNeighbors(n_neighbors=4, leaf_size=32, n_jobs=-1)
            # knn_1.fit(c1)
            # knn_2 = NearestNeighbors(n_neighbors=4, leaf_size=32, n_jobs=-1)
            # knn_2.fit(c2)

            # knn_dist_1 = np.mean(knn_1.kneighbors(c1)[0])
            # knn_dist_2 = np.mean(knn_2.kneighbors(c2)[0])

            # rad_dist_1 = pred_dist - dist_mat_1
            # rad_dist_1 = np.clip(rad_dist_1, a_min=0, a_max=None)
            # count_rad_1 = np.count_nonzero(rad_dist_1, axis=1)
            #
            # rad_dist_2 = pred_dist - dist_mat_2
            # rad_dist_2 = np.clip(rad_dist_2, a_min=0, a_max=None)
            # count_rad_2 = np.count_nonzero(rad_dist_2, axis=1)

            rad_dist_12_m = min_dist - dist_mat_12
            rad_dist_12_m = np.clip(rad_dist_12_m, a_min=0, a_max=None)
            count_rad_12_m = np.count_nonzero(rad_dist_12_m, axis=0)

            rad_dist_12_p = pred_dist - dist_mat_12
            rad_dist_12_p = np.clip(rad_dist_12_p, a_min=0, a_max=None)
            count_rad_12_p = np.count_nonzero(rad_dist_12_p, axis=0)

            # count_rad_1 = np.mean(count_rad_1)
            # count_rad_2 = np.mean(count_rad_2)
            count_rad_12_m = np.mean(count_rad_12_m)
            count_rad_12_p = np.mean(count_rad_12_p)

            # knn_coeff = knn_dist_2 / knn_dist_1
            # count_coeff = count_rad_2 / count_rad_1
            # edm_coeff = mean_dist_2 / mean_dist_1
            #
            # prec_coeff = (pred_dist / mean_dist_1) + 1
            # dist_coeff = (min_dist / mean_dist_1) + 1
            #
            # dens_coeff = (1 / (np.abs(1 - (count_coeff * edm_coeff)) + 1))
            # acc_coeff = (1 / (np.abs(1 - (prec_coeff * dist_coeff)) + 1))
            # fit_coeff = dens_coeff * acc_coeff

            return min_dist, pred_dist, count_rad_12_m, count_rad_12_p

        # def rad_count_dist(c1, c2, dist_metric='euclidean'):
        #     c1 = np.reshape(c1, (c1.shape[0], -1))
        #     c2 = np.reshape(c2, (c2.shape[0], -1))
        #
        #     min_dist = np.min(sp.spatial.distance.cdist(c1, c2, metric=dist_metric), axis=1)
        #     rad_dist = min_dist - sp.spatial.distance.cdist(c2, c2, metric=dist_metric)
        #     rad_dist = np.clip(rad_dist, a_min=0, a_max=None)
        #     count_rad = np.count_nonzero(rad_dist, axis=1)
        #     rad_dist = np.sum(rad_dist, axis=1)
        #
        #     rad_dist_comp = min_dist - sp.spatial.distance.cdist(c1, c1, metric=dist_metric)
        #     rad_dist_comp = np.clip(rad_dist_comp, a_min=0, a_max=None)
        #     count_rad_comp = np.count_nonzero(rad_dist_comp, axis=1)
        #     rad_dist_comp = np.sum(rad_dist_comp, axis=1)
        #
        #     min_dist = np.mean(min_dist)
        #     rad_dist = np.mean(rad_dist)
        #     count_rad = np.mean(count_rad)
        #
        #     rad_dist_comp = np.mean(rad_dist_comp)
        #     count_rad_comp = np.mean(count_rad_comp)
        #
        #     proj_score = (1 / (min_dist + 1)) * (1 / (rad_dist + 1))
        #     sym_rad_dist = (rad_dist + rad_dist_comp) / 2
        #     sym_proj_score = (1 / (min_dist + 1)) * (1 / (sym_rad_dist + 1))
        #
        #     return min_dist, rad_dist, count_rad, rad_dist_comp, count_rad_comp, proj_score, sym_rad_dist, sym_proj_score

        # import matplotlib.pyplot as plt
        #
        # fig, ax = plt.subplots()
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
        #     ax.scatter(xs, ys, marker='.', alpha=0.1, label=str(lab))
        # ax.legend()
        # ax.set_title('Projected samples (GT val classes)')
        # ax.grid(True)
        # fig.tight_layout()
        # plt.show(block=False)

        # fig, ax = plt.subplots()
        # ax.scatter(seq_tails_train_trans[:, 0], seq_tails_train_trans[:, 1], marker='.', alpha=0.1, label='train')
        # ax.legend()
        # ax.scatter(seq_tails_val_trans[:, 0], seq_tails_val_trans[:, 1], marker='.', alpha=0.1, label='val')
        # ax.legend()
        # ax.set_title('Projected samples (GT Splits)')
        # ax.grid(True)
        # fig.tight_layout()
        # plt.show(block=False)

        # print('KNN-dist: ' + ' '.join("%.4f" % x for x in compute_metrics(seq_tails_train_trans, seq_tails_val_trans, knn_train)))

        # train_probas = lda.predict_proba(np.reshape(seq_tails_train, (total_samples, -1)))
        # val_probas = lda.predict_proba(np.reshape(seq_tails_val, (total_samples, -1)))
        # print('KL: %f %f %f' % (np.mean(sp.stats.entropy(val_probas, train_probas)), np.mean(sp.stats.entropy(train_probas, val_probas)), np.mean(sp.stats.entropy(val_probas, val_probas))))

        for m, _ in enumerate(model_wraps):
            print(configs[m].save_path)

            gen_trans = lda_transform(gen_tails_val[m])
            # gen_probas = lda.predict_proba(np.reshape(gen_tails_val[m], (total_samples, -1)))

            # fig, ax = plt.subplots()
            # ax.scatter(seq_tails_val_trans[:, 0], seq_tails_val_trans[:, 1], marker='.', alpha=0.1, label='GT')
            # ax.legend()
            # ax.scatter(gen_trans[:, 0], gen_trans[:, 1], marker='.', alpha=0.1, label=configs[m].save_path)
            # ax.legend()
            # ax.set_title('Projected samples')
            # ax.grid(True)
            # fig.tight_layout()
            # plt.show(block=False)

            print('KNN-dist: ' + ' '.join("%.4f" % x for x in compute_metrics(seq_tails_val_trans, gen_trans)))
            # print('KL: %f %f' % (np.mean(sp.stats.entropy(val_probas, gen_probas)), np.mean(sp.stats.entropy(gen_probas, val_probas))))

        # plt.show()
        actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
                   'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking',
                   'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']

        gen_tails_val_trans = []
        for m, _ in enumerate(model_wraps):
            gen_tails_val_trans.append(lda_transform(gen_tails_val[m]))

        # for lab in sorted(set(labs_train)):
        #     print('\nAction: ' + actions[int(lab)])
        #     idxs = labs_val == lab
        #     trains_trans = seq_tails_train_trans[idxs, ...]
        #     vals_trans = seq_tails_val_trans[idxs, ...]

            # print('KNN-dist: ' + ' '.join("%.4f" % x for x in compute_metrics(trains_trans, vals_trans, knn_train)))

            # for m, _ in enumerate(model_wraps):
            #     print(configs[m].save_path)
            #
            #     vals_gen_trans = gen_tails_val_trans[m][idxs, ...]
            #
            #     print('KNN-dist: ' + ' '.join("%.4f" % x for x in compute_metrics(vals_trans, vals_gen_trans, knn_val)))



















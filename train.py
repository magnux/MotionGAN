from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from config import get_config
from data_input import DataInput
from utils.callbacks import TensorBoard
from models.motiongan import MotionGANV1, MotionGANV2, MotionGANV3, MotionGANV4
from utils.restore_keras_model import restore_keras_model
from tqdm import trange
from utils.viz import plot_seq_gif, plot_seq_emb
from utils.seq_utils import MASK_MODES, gen_mask, gen_latent_noise

def _reset_rand_seed():
    seed = 42
    np.random.seed(seed)
    tf.set_random_seed(seed)


logging = tf.logging
flags = tf.flags
flags.DEFINE_bool("verbose", False, "To talk or not to talk")
flags.DEFINE_string("save_path", None, "Model output directory")
flags.DEFINE_string("config_file", None, "Model config file")
FLAGS = flags.FLAGS

if __name__ == "__main__":
    _reset_rand_seed()

    if not tf.gfile.Exists('./save'):
        tf.gfile.MkDir('./save')

    # Config stuff
    config = get_config(FLAGS)

    data_input = DataInput(config)
    _reset_rand_seed()  # Creating data_input object might introduce some randomness
    train_batches = data_input.train_epoch_size
    train_generator = data_input.batch_generator(True)
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

    def save_models():
        model_wrap.disc_model.save(config.save_path + '_disc_weights.hdf5')
        model_wrap.gen_model.save(config.save_path + '_gen_weights.hdf5')

    if config.epoch > 0:
        model_wrap.disc_model = restore_keras_model(
            model_wrap.disc_model, config.save_path + '_disc_weights.hdf5', False)
        model_wrap.gen_model = restore_keras_model(
            model_wrap.gen_model, config.save_path + '_gen_weights.hdf5', False)
    else:
        save_models()
        config.save()

    tensorboard = TensorBoard(log_dir=config.save_path + '_logs',
                              epoch=config.epoch,
                              n_batches=train_batches,
                              batch_size=config.batch_size,
                              write_graph=True)
    tensorboard.set_model(model_wrap.gan_model)

    try:
        while config.epoch < config.num_epochs:
            tensorboard.on_epoch_begin(config.epoch)

            if config.lr_decay:
                learning_rate = config.learning_rate * (0.1 ** (config.epoch // (config.num_epochs // 3)))
                # learning_rate = config.learning_rate * (1.0 - (config.epoch / config.num_epochs))
                model_wrap.update_lr(learning_rate)

            t = trange(config.batch, train_batches)
            t.set_description('| ep: %d | lr: %.2e |' % (config.epoch, learning_rate))
            disc_loss_sum = 0.0
            gen_loss_sum = 0.0
            keep_prob = 0.8 - (0.6 * config.epoch / config.num_epochs)
            for batch in t:
                tensorboard.on_batch_begin(batch)

                disc_batches = 5
                # disc_batches = 55 if ((config.epoch < 1 and batch < train_batches // 10)
                #                           or (batch % 10 == 0)) else 5
                disc_loss = 0.0
                loss_real = 0.0
                loss_fake = 0.0
                for disc_batch in range(disc_batches):
                    labs_batch, poses_batch = train_generator.next()

                    mask_batch = poses_batch[..., 3, np.newaxis]
                    mask_batch = mask_batch * gen_mask(np.random.randing(len(MASK_MODES)), keep_prob, config.batch_size,
                                                       config.njoints, model_wrap.seq_len, config.body_members)
                    poses_batch = poses_batch[..., :3]

                    disc_inputs = [poses_batch]
                    gen_inputs = [poses_batch, mask_batch]
                    place_holders = []
                    if config.action_cond:
                        place_holders.append(labs_batch[:, 2])
                    if config.latent_cond_dim > 0:
                        gen_inputs.append(gen_latent_noise(config.batch_size, config.latent_cond_dim))

                    losses = model_wrap.disc_train(disc_inputs + gen_inputs + place_holders)

                    if disc_batch == 0:
                        disc_losses = losses
                    else:
                        for key in disc_losses.keys():
                            disc_losses[key] += losses[key]

                for key in disc_losses.keys():
                    disc_losses[key] /= disc_batches

                labs_batch, poses_batch = train_generator.next()

                mask_batch = poses_batch[..., 3, np.newaxis]
                mask_batch = mask_batch * gen_mask(np.random.randing(len(MASK_MODES)), keep_prob, config.batch_size,
                                                   config.njoints, model_wrap.seq_len, config.body_members)
                poses_batch = poses_batch[..., :3]

                gen_inputs = [poses_batch, mask_batch]
                place_holders = []
                if config.action_cond:
                    place_holders.append(labs_batch[:, 2])
                if config.latent_cond_dim > 0:
                    gen_inputs.append(gen_latent_noise(config.batch_size, config.latent_cond_dim))

                gen_losses = model_wrap.gen_train(gen_inputs + place_holders)

                # Output to terminal, note output is averaged over the epoch
                disc_loss_sum += disc_losses['train/disc_loss_wgan']
                gen_loss_sum += gen_losses['train/gen_loss_wgan']
                t.set_postfix(disc_loss='%.2e' % (disc_loss_sum / (batch + 1)),
                              gen_loss='%.2e' % (gen_loss_sum / (batch + 1)))

                logs = disc_losses.copy()
                logs.update(gen_losses)

                # Check for a bad minima, leading to nan weights
                if True in [np.isnan(log) for log in logs.values()]:
                    for log_key, log_val in logs.items():
                        if np.isnan(log_val):
                            print('nans found in %s' % log_key)
                    print('restarting epoch')
                    config.nan_restarts += 1
                    assert config.nan_restarts < 10, "restarted too many times because of nans"
                    break

                tensorboard.on_batch_end(batch, logs)

                config.batch = batch + 1

            # Restarting epoch after sudden break
            if config.batch < train_batches:
                model_wrap.disc_model = restore_keras_model(
                    model_wrap.disc_model, config.save_path + '_disc_weights.hdf5', False)
                model_wrap.gen_model = restore_keras_model(
                    model_wrap.gen_model, config.save_path + '_gen_weights.hdf5', False)
                config.batch = 0
                continue

            labs_batch, poses_batch = val_generator.next()

            mask_batch = poses_batch[..., 3, np.newaxis]
            mask_mode = np.random.randing(len(MASK_MODES))
            mask_batch = mask_batch * gen_mask(mask_mode, keep_prob, config.batch_size,
                                               config.njoints, model_wrap.seq_len, config.body_members)
            poses_batch = poses_batch[..., :3]

            disc_inputs = [poses_batch]
            gen_inputs = [poses_batch, mask_batch]
            place_holders = []
            if config.action_cond:
                place_holders.append(labs_batch[:, 2])
            if config.latent_cond_dim > 0:
                gen_inputs.append(gen_latent_noise(config.batch_size, config.latent_cond_dim))

            disc_losses = model_wrap.disc_eval(disc_inputs + gen_inputs + place_holders)
            gen_losses = model_wrap.gen_eval(gen_inputs + place_holders)
            if config.use_pose_fae:
                fae_z = gen_losses.pop('fae_z', None)
            # aux_out = gen_losses.pop('aux_out', None)
            gen_outputs = gen_losses.pop('gen_outputs', None)

            logs = disc_losses.copy()
            logs.update(gen_losses)

            # Generating images
            if (config.epoch % (config.num_epochs // 10)) == 0 or config.epoch == (config.num_epochs - 1):
                if config.normalize_data:
                    poses_batch = data_input.unnormalize_poses(poses_batch)
                    gen_outputs = data_input.unnormalize_poses(gen_outputs)
                    # aux_out = data_input.unnormalize_poses(aux_out)
                for i in range(10):  # config.batch_size
                    gif_name = '%s_tmp.gif' % config.save_path
                    gif_height, gif_width = plot_seq_gif(
                        np.concatenate([poses_batch[np.newaxis, i, ...],
                                        # aux_out[np.newaxis, i, ...],
                                        gen_outputs[np.newaxis, i, ...]]),
                        labs_batch[i, ...],
                        config.data_set,
                        seq_masks=mask_batch[i, ...],
                        extra_text='mask mode: %s' % MASK_MODES[mask_mode],
                        save_path=gif_name)

                    with open(gif_name, 'rb') as f:
                        encoded_image_string = f.read()

                    logs['custom_img_%d' % i] = {'height': gif_height,
                                                 'width': gif_width,
                                                 'enc_string': encoded_image_string}

                    if config.use_pose_fae:
                        png_name = '%s_mask_tmp.png' % config.save_path
                        plot_seq_emb(fae_z[i, ...], png_name)

                        with open(png_name, 'rb') as f:
                            encoded_image_string = f.read()

                        logs['custom_img_emb_%d' % i] = {'height': int(fae_z.shape[1]),
                                                         'width': int(fae_z.shape[2]),
                                                         'enc_string': encoded_image_string}

            tensorboard.on_epoch_end(config.epoch, logs)

            config.epoch += 1
            config.batch = 0

            save_models()
            config.save()

    except KeyboardInterrupt:
        save_models()
        config.save()

    tensorboard.on_train_end()

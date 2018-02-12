from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from config import get_config
from data_input import DataInput
from utils.callbacks import TensorBoard
from models.motiongan import MotionGANV1, MotionGANV2
from utils.restore_keras_model import restore_keras_model
from tqdm import trange
from utils.viz import plot_gif

logging = tf.logging
flags = tf.flags
flags.DEFINE_bool("verbose", False, "To talk or not to talk")
flags.DEFINE_string("save_path", None, "Model output directory")
flags.DEFINE_string("config_file", None, "Model config file")
FLAGS = flags.FLAGS

if __name__ == "__main__":
    if not tf.gfile.Exists('./save'):
        tf.gfile.MkDir('./save')

    # Config stuff
    config = get_config(FLAGS)

    data_input = DataInput(config)
    train_batches = data_input.train_epoch_size
    train_generator = data_input.batch_generator(True)

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

    if config.epoch > 0:
        model_wrap.disc_model = restore_keras_model(
            model_wrap.disc_model, config.save_path + '_disc_weights.hdf5', False)
        model_wrap.gen_model = restore_keras_model(
            model_wrap.gen_model, config.save_path + '_gen_weights.hdf5', False)

    tensorboard = TensorBoard(log_dir=config.save_path + '_logs',
                              batch_size=config.batch_size, write_graph=True)
    tensorboard.set_model(model_wrap.gan_model)

    def gen_latent_noise():
        return np.random.uniform(size=(config.batch_size, config.latent_cond_dim))

    for epoch in range(config.epoch, config.num_epochs):
        tensorboard.on_epoch_begin(epoch)

        if config.lr_decay:
            # learning_rate = config.learning_rate * (0.1 ** (epoch // (config.num_epochs // 3)))
            learning_rate = config.learning_rate * (1.0 - (epoch / config.num_epochs))
            model_wrap.update_lr(learning_rate)

        t = trange(train_batches)
        t.set_description('| ep: %d | lr: %.2e |' % (epoch, learning_rate))
        disc_loss_sum = 0
        loss_real_sum = 0
        loss_fake_sum = 0
        gen_loss_sum = 0
        for batch_num in t:
            disc_batches = 55 if ((epoch < 1 and batch_num < train_batches // 10)
                                      or (batch_num % 10 == 0)) else 5
            disc_loss = 0
            loss_real = 0
            loss_fake = 0
            for _ in range(disc_batches):
                labs_batch, poses_batch = train_generator.next()

                disc_inputs = [poses_batch]
                gen_inputs = [poses_batch]
                place_holders = [True]  # disc_training is true
                if config.action_cond:
                    place_holders.append(labs_batch[:, 2])
                if config.latent_cond_dim > 0:
                    latent_noise = gen_latent_noise()
                    gen_inputs.append(latent_noise)

                disc_losses = model_wrap.disc_train(disc_inputs + gen_inputs + place_holders)
                disc_loss += disc_losses[0]
                loss_real += disc_losses[1]
                loss_fake += disc_losses[2]

            disc_loss_sum += (disc_loss / disc_batches)
            loss_real_sum += (loss_real / disc_batches)
            loss_fake_sum += (loss_fake / disc_batches)

            labs_batch, poses_batch = train_generator.next()

            gen_inputs = [poses_batch]
            place_holders = [False]  # disc_training is false, so gen_training is True
            if config.action_cond:
                place_holders.append(labs_batch[:, 2])
            if config.latent_cond_dim > 0:
                latent_noise = gen_latent_noise()
                gen_inputs.append(latent_noise)

            gen_loss = model_wrap.gen_train(gen_inputs + place_holders)

            gen_loss_sum += gen_loss[0]
            t.set_postfix(disc_loss='%.2e' % (disc_loss_sum / (batch_num + 1)),
                          gen_loss='%.2e' % (gen_loss_sum / (batch_num + 1)))

            logs = {
                'disc_loss': (disc_loss / disc_batches),
                'loss_real': (loss_real / disc_batches),
                'loss_fake': (loss_fake / disc_batches),
                'gen_loss': gen_loss[0]
            }

            tensorboard.on_batch_end(batch_num, logs)

        model_wrap.disc_model.save(config.save_path + '_disc_weights.hdf5')
        model_wrap.gen_model.save(config.save_path + '_gen_weights.hdf5')

        config.epoch = epoch + 1
        config.save()

        # Generating images and logging
        gen_outputs = model_wrap.gen_model.predict(gen_inputs, config.batch_size)

        logs = {
            'disc_loss': disc_loss_sum / train_batches,
            'loss_real': loss_real_sum / train_batches,
            'loss_fake': loss_fake_sum / train_batches,
            'gen_loss': gen_loss_sum / train_batches
        }

        for i in range(config.batch_size):
            gif_name = '%s_tmp.gif' % config.save_path
            gif_height, gif_width = plot_gif(poses_batch[i, ...],
                                             gen_outputs[i, ...],
                                             labs_batch[i, ...], gif_name)

            with open(gif_name, 'rb') as f:
                encoded_image_string = f.read()

            logs['custom_img_%d' % i] = {'height': gif_height,
                                         'width': gif_width,
                                         'enc_string': encoded_image_string}

        tensorboard.on_epoch_end(epoch + 1, logs)

    tensorboard.on_train_end(None)

from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from config import get_config
from data_input import DataInput
from utils.callbacks import TensorBoard
from tensorflow.contrib.keras.api.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from models.posevae import PoseVAEV1
from utils.restore_keras_model import restore_keras_model
from utils.threadsafe_iter import threadsafe_generator

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
    val_batches = data_input.val_epoch_size
    val_generator = data_input.batch_generator(False)

    model_wrap = PoseVAEV1(config)

    if FLAGS.verbose:
        print('VAE model:')
        print(model_wrap.encoder.summary())
        print(model_wrap.decoder.summary())

    if config.epoch > 0:
        model_wrap.autoencoder = restore_keras_model(
            model_wrap.autoencoder, config.save_path + '_weights.hdf5')

    # Callbacks
    def schedule(epoch):
        config.epoch = epoch
        config.save()
        learning_rate = config.learning_rate * (
                0.1 ** (epoch // (config.num_epochs // 3)))
        # print('LR: %.2e' % learning_rate)
        return learning_rate


    lr_scheduler = LearningRateScheduler(schedule)
    checkpointer = ModelCheckpoint(filepath=config.save_path + '_weights.hdf5')
    tensorboard = TensorBoard(log_dir=config.save_path + '_logs',
                              epoch=config.epoch,
                              n_batches=train_batches,
                              batch_size=config.batch_size,
                              write_graph=True)
    tensorboard.set_model(model_wrap.autoencoder)

    @threadsafe_generator
    def vae_train_generator():
        while True:
            vae_epsilon = np.random.normal(size=(config.batch_size, model_wrap.seq_len, model_wrap.vae_latent_dim), loc=0., scale=1.0)
            _, train_batch = train_generator.next()
            yield ([train_batch, vae_epsilon], train_batch)

    @threadsafe_generator
    def vae_val_generator():
        while True:
            vae_epsilon = np.zeros(shape=(config.batch_size, model_wrap.seq_len, model_wrap.vae_latent_dim))
            _, val_batch = val_generator.next()
            yield ([val_batch, vae_epsilon], val_batch)

    # Training call
    model_wrap.autoencoder.fit_generator(generator=vae_train_generator(), steps_per_epoch=train_batches*50,
                                         validation_data=vae_val_generator(), validation_steps=1,
                                         epochs=config.num_epochs,
                                         max_q_size=config.batch_size * 8, workers=2,
                                         verbose=1 if FLAGS.verbose else 2,
                                         callbacks=[lr_scheduler, checkpointer, tensorboard])
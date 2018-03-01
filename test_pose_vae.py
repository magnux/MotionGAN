from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from config import get_config
from data_input import DataInput
from utils.callbacks import TensorBoard
from tensorflow.contrib.keras.api.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from models.posevae import PoseVAEV1
from utils.restore_keras_model import restore_keras_model
from utils.viz import plot_gif

logging = tf.logging
flags = tf.flags
flags.DEFINE_bool("verbose", False, "To talk or not to talk")
flags.DEFINE_string("save_path", None, "Model output directory")
flags.DEFINE_string("config_file", None, "Model config file")
FLAGS = flags.FLAGS

if __name__ == "__main__":

    # Config stuff
    config = get_config(FLAGS)

    data_input = DataInput(config)
    val_batches = data_input.val_epoch_size
    val_generator = data_input.batch_generator(False)

    model_wrap = PoseVAEV1(config)

    if FLAGS.verbose:
        print('VAE model:')
        print(model_wrap.encoder.summary())
        print(model_wrap.decoder.summary())

    model_wrap.autoencoder = restore_keras_model(
        model_wrap.autoencoder, config.save_path + '_weights.hdf5')

    labs_batch, val_batch = val_generator.next()
    vae_epsilon = np.zeros(shape=(config.batch_size, model_wrap.seq_len, model_wrap.vae_latent_dim))

    # Training call
    predictions = model_wrap.autoencoder.predict([val_batch, vae_epsilon], config.batch_size)

    test_path = config.save_path + "_test"
    if not tf.gfile.Exists(test_path):
        tf.gfile.MkDir(test_path)

    for i in range(config.batch_size):
        plot_gif(val_batch[i, ...], predictions[i, ...], labs_batch[i, ...], config.data_set, test_path + ("/%03d.gif" % i))
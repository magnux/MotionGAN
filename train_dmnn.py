from __future__ import division
import tensorflow as tf
from config import get_config
from data_input import DataInput
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from utils.callbacks import TensorBoard
from models.dmnn import DMNNv1
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
    train_batches = data_input.train_epoch_size * config.epoch_factor
    val_batches = data_input.val_epoch_size

    # Model building
    if config.model_type == 'dmnn':
        if config.model_version == 'v1':
            model_wrap = DMNNv1(config)

    if FLAGS.verbose:
        print('DMNN model:')
        print(model_wrap.model.summary())

    if config.epoch > 0:
        model_wrap.model = restore_keras_model(model_wrap.model, config.save_path + '_weights.hdf5')

    # Callbacks
    def schedule(epoch):
        config.epoch = epoch
        config.save()
        learning_rate = config.learning_rate * (0.1 ** (epoch // (config.num_epochs // 3)))
        print('LR: %.2e' % learning_rate)
        return learning_rate
    lr_scheduler = LearningRateScheduler(schedule)
    checkpointer = ModelCheckpoint(filepath=config.save_path + '_weights.hdf5')
    tensorboard = TensorBoard(log_dir=config.save_path + '_logs',
                              epoch=config.epoch,
                              n_batches=train_batches,
                              batch_size=config.batch_size)


    @threadsafe_generator
    def batch_generator(is_training):
        generator = data_input.batch_generator(is_training)
        while True:
            labs_batch, poses_batch = generator.next()
            yield (poses_batch[..., :3], labs_batch[:, 2])


    # Training call
    model_wrap.model.fit_generator(generator=batch_generator(True),
                                   steps_per_epoch=train_batches, epochs=config.num_epochs, initial_epoch=config.epoch,
                                   validation_data=batch_generator(False), validation_steps=val_batches,
                                   max_q_size=config.batch_size * 8, workers=2,
                                   verbose=1 if FLAGS.verbose else 2, callbacks=[lr_scheduler, checkpointer, tensorboard])

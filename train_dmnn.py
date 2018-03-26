from __future__ import division
import tensorflow as tf
import numpy as np
from config import get_config
from data_input import DataInput
from utils.callbacks import TensorBoard
from models.dmnn import DMNNv1
from models.motiongan import MotionGANV1, MotionGANV2, MotionGANV3, MotionGANV4
from utils.restore_keras_model import restore_keras_model
from tqdm import trange

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

    if config.motiongan_save_path is not None:
        FLAGS.save_path = config.motiongan_save_path
        FLAGS.config_file = None

        motiongan_config = get_config(FLAGS)
        motiongan_config.batch_size = config.batch_size

        # Model building
        if motiongan_config.model_type == 'motiongan':
            if motiongan_config.model_version == 'v1':
                motiongan_model_wrap = MotionGANV1(motiongan_config)
            if motiongan_config.model_version == 'v2':
                motiongan_model_wrap = MotionGANV2(motiongan_config)
            if motiongan_config.model_version == 'v3':
                motiongan_model_wrap = MotionGANV3(motiongan_config)
            if motiongan_config.model_version == 'v4':
                motiongan_model_wrap = MotionGANV4(motiongan_config)

        if FLAGS.verbose:
            print('Generator model:')
            print(motiongan_model_wrap.gen_model.summary())

        assert motiongan_config.epoch > 0, 'untrained model'

        motiongan_model_wrap.gen_model = restore_keras_model(
            motiongan_model_wrap.gen_model, motiongan_config.save_path + '_gen_weights.hdf5', False)

        gen_model = motiongan_model_wrap.gen_model
        
        del motiongan_model_wrap.disc_model
        del motiongan_model_wrap.gan_model
        del motiongan_model_wrap


    tensorboard = TensorBoard(log_dir=config.save_path + '_logs',
                              epoch=config.epoch,
                              n_batches=train_batches,
                              batch_size=config.batch_size)
    tensorboard.set_model(model_wrap.model)

    train_generator = data_input.batch_generator(True)
    val_generator = data_input.batch_generator(False)

    def save_models():
        model_wrap.model.save(config.save_path + '_weights.hdf5')

    def run_epoch(training, learning_rate):
        batches = train_batches if training else val_batches
        generator = train_generator if training else val_generator

        tensorboard.on_epoch_begin(config.epoch)

        t = trange(batches)
        t.set_description('%s| ep: %d | lr: %.2e |' %
                          ('| train ' if training else '|  val  ',
                           config.epoch, learning_rate))
        loss_sum = 0
        acc_sum = 0
        for batch in t:
            tensorboard.on_batch_begin(batch)
            labs_batch, poses_batch = generator.next()
            labs_batch = labs_batch[:, 2]
            mask_batch = poses_batch[..., 3, np.newaxis]
            poses_batch = poses_batch[..., :3]

            run_on_batch = model_wrap.model.train_on_batch if training else\
                           model_wrap.model.test_on_batch
            loss, acc = run_on_batch(poses_batch, labs_batch)
            loss_sum += loss
            acc_sum += acc

            if config.motiongan_save_path is not None and training:
                if motiongan_config.normalize_data:
                    poses_batch = data_input.normalize_poses(poses_batch)

                gen_inputs = [poses_batch, mask_batch]
                gen_batch = gen_model.predict_on_batch(gen_inputs)

                if motiongan_config.normalize_data:
                    gen_batch = data_input.denormalize_poses(gen_batch)

                model_wrap.model.train_on_batch(gen_batch, labs_batch)

            t.set_postfix(loss='%.2e' % (loss_sum / (batch + 1)),
                          acc='%.2f' % (acc_sum * 100 / (batch + 1)))

            prefix = 'train' if training else 'val'
            logs = {prefix + '/loss': loss, prefix + '/acc': acc}
            tensorboard.on_batch_end(batch, logs)

        tensorboard.on_epoch_end(config.epoch)

    while config.epoch < config.num_epochs:

        if config.lr_decay:
            if config.epoch == 0:
                learning_rate = config.learning_rate * 0.1
            else:
                learning_rate = config.learning_rate * \
                                (0.1 ** (config.epoch // (config.num_epochs // 3)))
            model_wrap.update_lr(learning_rate)

        run_epoch(True, learning_rate)
        run_epoch(False, learning_rate)

        save_models()
        config.epoch += 1
        config.save()

    tensorboard.on_train_end()

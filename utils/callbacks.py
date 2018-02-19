from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.callbacks import Callback
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary as tf_summary
import tensorflow as tf

class ReduceLROnPercentPlateau(Callback):

    def __init__(self, monitor='val_loss', factor=0.1, patience=1,
                 verbose=0, mode='auto', pct_epsilon=1, cooldown=0, min_lr=1e-6):
        super(ReduceLROnPercentPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.pct_epsilon = pct_epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.greater(b - a, (b / 100) * self.pct_epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.less(b - a, (b / 100) * self.pct_epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0
        self.lr_epsilon = self.min_lr * 1e-4

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Learning Rate Plateau Reducing requires %s available!' %
                          self.monitor, RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr + self.lr_epsilon:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: reducing learning rate to %s.' % (epoch, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0


class TensorBoard(Callback):
    # pylint: disable=line-too-long
    """Tensorboard basic visualizations.

    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.

    TensorBoard is a visualization tool provided with TensorFlow.

    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:

    ```sh
    tensorboard --logdir=/full_path_to_your_logs
    ```

    You can find more information about TensorBoard
    [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

    Arguments:
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
    """

    # pylint: enable=line-too-long

    def __init__(self,
                 log_dir='./logs',
                 histogram_freq=0,
                 epoch=0,
                 n_batches=0,
                 batch_size=0,
                 write_graph=True,
                 write_grads=False,
                 write_images=False):
        super(TensorBoard, self).__init__()
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.epoch = epoch
        self.n_batches = n_batches
        self.batch_size = batch_size

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:
                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    tf_summary.histogram(mapped_weight_name, weight)
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)

                        def is_indexed_slices(grad):
                            return type(grad).__name__ == 'IndexedSlices'

                        grads = [
                            grad.values if is_indexed_slices(grad) else grad
                            for grad in grads]
                        tf_summary.histogram(
                            '{}_grad'.format(mapped_weight_name), grads)
                    if self.write_images:
                        w_img = array_ops.squeeze(weight)
                        shape = K.int_shape(w_img)
                        if len(shape) == 2:  # dense layer kernel case
                            if shape[0] > shape[1]:
                                w_img = array_ops.transpose(w_img)
                                shape = K.int_shape(w_img)
                            w_img = array_ops.reshape(w_img,
                                                      [1, shape[0], shape[1], 1])
                        elif len(shape) == 3:  # convnet case
                            if K.image_data_format() == 'channels_last':
                                # switch to channels_first to display
                                # every kernel as a separate image
                                w_img = array_ops.transpose(w_img, perm=[2, 0, 1])
                                shape = K.int_shape(w_img)
                            w_img = array_ops.reshape(w_img,
                                                      [shape[0], shape[1], shape[2], 1])
                        elif len(shape) == 1:  # bias case
                            w_img = array_ops.reshape(w_img, [1, shape[0], 1, 1])
                        else:
                            # not possible to handle 3D convnets etc.
                            continue

                        shape = K.int_shape(w_img)
                        assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                        tf_summary.image(mapped_weight_name, w_img)

                if hasattr(layer, 'output'):
                    tf_summary.histogram('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf_summary.merge_all()

        if self.write_graph:
            self.writer = tf_summary.FileWriter(self.log_dir, self.sess.graph)
        else:
            self.writer = tf_summary.FileWriter(self.log_dir)

    def _current_step(self):
        return (self.epoch * self.n_batches) + self.batch

    def _save_logs(self, logs):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            elif 'custom_img' in name:
                self._save_custom_img(name, value)
            else:
                self._save_scalar(name, value)
        self.writer.flush()

    def _save_scalar(self, name, value):
        summary = tf_summary.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value if isinstance(value, float) else value.item()
        summary_value.tag = name
        self.writer.add_summary(summary, self._current_step())

    def _save_custom_img(self, name, value):
        summary = tf_summary.Summary()
        image = tf.Summary.Image()
        image.height = value['height']
        image.width = value['width']
        image.colorspace = 3  # code for 'RGB'
        image.encoded_image_string = value['enc_string']
        summary.value.add(tag=name, image=image)
        self.writer.add_summary(summary, self._current_step())

    def on_batch_begin(self, batch, logs=None):
        self.batch = batch

    def on_batch_end(self, batch, logs=None):
        self.batch = batch
        self._save_logs(logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.batch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        self.batch = self.n_batches - 1

        if not self.validation_data and self.histogram_freq:
            raise ValueError('If printing histograms, validation_data must be '
                             'provided, and cannot be a generator.')
        if self.validation_data and self.histogram_freq:
            if self.epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (
                        self.model.inputs + self.model.targets + self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    batch_val = []
                    batch_val.append(val_data[0][i:i + step])
                    batch_val.append(val_data[1][i:i + step])
                    batch_val.append(val_data[2][i:i + step])
                    if self.model.uses_learning_phase:
                        # do not slice the learning phase
                        batch_val = [x[i:i + step] for x in val_data[:-1]]
                        batch_val.append(val_data[-1])
                    else:
                        batch_val = [x[i:i + step] for x in val_data]
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, self.epoch * self.batch_size)
                    i += self.batch_size

        self._save_logs(logs)

    def on_train_end(self, logs=None):
        self.writer.close()

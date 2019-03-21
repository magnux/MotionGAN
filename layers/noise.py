from tensorflow.contrib.keras.api.keras.layers import Layer
from tensorflow.contrib.keras.api.keras import backend as K


class GaussianMultNoise(Layer):
    """Apply multiplicative zero-centered Gaussian noise.

    Arguments:
        stddev: float, standard deviation of the noise distribution.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.
    """

    def __init__(self, stddev, **kwargs):
        super(GaussianMultNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        def noised():
            return inputs + (inputs * K.random_normal(shape=K.shape(inputs), mean=0., stddev=self.stddev))

        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(GaussianMultNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NoiseInjection(Layer):
    """Apply additive Gaussian Noise, with learned weights

    Arguments:
        stddev: float, standard deviation of the noise distribution.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.
    """

    def __init__(self, stddev=1.0, **kwargs):
        super(NoiseInjection, self).__init__(**kwargs)
        self.stddev = stddev

    def build(self, input_shape):
        channels = input_shape[-1]
        self.noise_w = self.add_weight(shape=(1, 1, 1, channels), name='noise_w', initializer='glorot_uniform')

    def call(self, inputs, seed=None):
        return inputs + (self.noise_w * K.random_normal(shape=K.shape(inputs), mean=0., stddev=self.stddev, seed=seed))

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(NoiseInjection, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras.engine import Layer
from tensorflow.python.keras._impl.keras.layers.recurrent import RNN

class RelationalMemoryCell(Layer):
    """Constructs a `RelationalMemory` Cell.
    Args:
      mem_slots: The total number of memory slots to use.
      head_size: The size of an attention head.
      num_heads: The number of attention heads to use. Defaults to 1.
      forget_bias: Bias to use for the forget gate, assuming we are using
        some form of gating. Defaults to 1.
      input_bias: Bias to use for the input gate, assuming we are using
        some form of gating. Defaults to 0.
      gate_style: Whether to use per-element gating ('unit'),
        per-memory slot gating ('memory'), or no gating at all (None).
        Defaults to `unit`.
      attention_mlp_layers: Number of layers to use in the post-attention
        MLP. Defaults to 2.
      key_size: Size of vector to use for key & query vectors in the attention
        computation. Defaults to None, in which case we use `head_size`.
    """

    def __init__(self, mem_slots, head_size, num_heads=1,
                 forget_bias=1.0, input_bias=0.0, gate_style='memory',
                 attention_mlp_layers=2, key_size=None, **kwargs):
        super(RelationalMemoryCell, self).__init__(**kwargs)
        self.mem_slots = mem_slots  # Denoted as N.
        self.head_size = head_size
        self.num_heads = num_heads  # Denoted as H.
        self.mem_size = self.head_size * self.num_heads
        self.key_size = key_size if key_size else self.head_size
        self.value_size = self.head_size  # Denoted as V.
        self.qkv_size = 2 * self.key_size + self.value_size
        self.total_size = self.qkv_size * self.num_heads  # Denote as F.

        self.forget_bias = forget_bias
        self.input_bias = input_bias

        if gate_style not in ['unit', 'memory', None]:
            raise ValueError(
                'gate_style must be one of [\'unit\', \'memory\', None]. Got: '
                '{}.'.format(gate_style))
        self.gate_style = gate_style
        self.num_gates = 2 * self._calculate_gate_size()

        if attention_mlp_layers < 1:
            raise ValueError('attention_mlp_layers must be >= 1. Got: {}.'.format(
                attention_mlp_layers))
        self.attention_mlp_layers = attention_mlp_layers

        self.state_size = self.mem_slots * self.mem_size


    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel_qkv = self.add_weight(
            shape=(self.mem_size, self.total_size),
            name='kernel_qkv',
            initializer='glorot_uniform')

        self.bias_qkv = self.add_weight(
            shape=(self.total_size,),
            name='bias_qkv',
            initializer='zeros')

        self.kernel_gi = self.add_weight(
            shape=(self.mem_size, self.num_gates),
            name='kernel_gi',
            initializer='glorot_uniform')

        self.bias_gi = self.add_weight(
            shape=(self.num_gates,),
            name='bias_gi',
            initializer='glorot_uniform')

        self.kernel_gm = self.add_weight(
            shape=(self.mem_size, self.num_gates),
            name='kernel_gm',
            initializer='glorot_uniform')

        self.bias_gm = self.add_weight(
            shape=(self.num_gates,),
            name='bias_gm',
            initializer='glorot_uniform')

        self.kernel_in = self.add_weight(
            shape=(input_dim, self.mem_size),
            name='kernel_in',
            initializer='glorot_uniform')

        self.bias_in = self.add_weight(
            shape=(self.mem_size,),
            name='bias_in',
            initializer='glorot_uniform')

        self.mlp_kernels = []
        self.mlp_biases = []
        for i in range(self.attention_mlp_layers):
            mlp_kernel = self.add_weight(
                shape=(self.mem_size, self.mem_size),
                name='mlp_kernel_%d' % i,
                initializer='glorot_uniform')
            mlp_bias = self.add_weight(
                shape=(self.mem_size,),
                name='mlp_bias_%d' % i,
                initializer='glorot_uniform')
            self.mlp_kernels.append(mlp_kernel)
            self.mlp_biases.append(mlp_bias)

        # self.offset_qkv = self.add_weight(
        #     shape=(self.total_size,),
        #     name='offset_qkv',
        #     initializer='zeros')
        #
        # self.scale_qkv = self.add_weight(
        #     shape=(self.total_size,),
        #     name='scale_qkv',
        #     initializer='ones')
        #
        # self.offset_mem_0 = self.add_weight(
        #     shape=(self.mem_size,),
        #     name='offset_mem_0',
        #     initializer='zeros')
        #
        # self.scale_mem_0 = self.add_weight(
        #     shape=(self.mem_size,),
        #     name='scale_mem_0',
        #     initializer='ones')
        #
        # self.offset_mem_1 = self.add_weight(
        #     shape=(self.mem_size,),
        #     name='offset_mem_1',
        #     initializer='zeros')
        #
        # self.scale_mem_1 = self.add_weight(
        #     shape=(self.mem_size,),
        #     name='scale_mem_1',
        #     initializer='ones')

        self.built = True

    def _calculate_gate_size(self):
        """Calculate the gate size from the gate_style.
        Returns:
          The per sample, per head parameter size of each gate.
        """
        if self.gate_style == 'unit':
            return self.num_heads * self.head_size
        elif self.gate_style == 'memory':
            return 1
        else:  # self._gate_style == None
            return 0

    def call(self, inputs, memory, training=None):
        batch_size = int(inputs.shape[0])

        memory = K.reshape(memory, (batch_size, self.mem_slots, self.mem_size))
        inputs = self._linear(inputs, self.kernel_in, self.bias_in)
        inputs_reshape = K.expand_dims(inputs, axis=1)

        memory_plus_input = K.concatenate([memory, inputs_reshape], axis=1)
        next_memory = self._attend_over_memory(memory_plus_input)

        n = inputs_reshape.get_shape().as_list()[1]
        next_memory = next_memory[:, :-n, :]

        input_gate, forget_gate = self._create_gates(inputs_reshape, memory)
        next_memory = input_gate * K.tanh(next_memory)
        next_memory += forget_gate * memory
        next_memory = K.batch_flatten(next_memory)

        return next_memory, (next_memory,)

    def _attend_over_memory(self, memory):
        """Perform multiheaded attention over `memory`.
        Args:
          memory: Current relational memory.
        Returns:
          The attended-over memory.
        """

        # Add a skip connection to the multiheaded attention's input.
        memory = memory + self._multihead_attention(memory)
        # memory = self._layer_norm(memory, self.offset_mem_0, self.scale_mem_0)

        # Add a skip connection to the attention_mlp's input.
        memory = memory + self._attention_mlp(memory)
        # memory = self._layer_norm(memory, self.offset_mem_1, self.scale_mem_1)

        return memory

    def _multihead_attention(self, memory):
        """Perform multi-head attention from 'Attention is All You Need'.
            Implementation of the attention mechanism from
            https://arxiv.org/abs/1706.03762.
            Args:
              memory: Memory tensor to perform attention on.
            Returns:
              new_memory: New memory tensor.
            """

        batch_size = int(memory.shape[0])

        qkv = self._linear(memory, self.kernel_qkv, self.bias_qkv)
        # qkv = self._layer_norm(qkv, self.offset_qkv, self.scale_qkv)

        mem_slots = memory.get_shape().as_list()[1]  # Denoted as N.

        # [B, N, F] -> [B, N, H, F/H]
        qkv_reshape = K.reshape(qkv, (batch_size, mem_slots, self.num_heads, self.qkv_size))

        # [B, N, H, F/H] -> [B, H, N, F/H]
        qkv_transpose = K.permute_dimensions(qkv_reshape, [0, 2, 1, 3])
        q, k, v = tf.split(qkv_transpose, [self.key_size, self.key_size, self.value_size], -1)

        q *= self.qkv_size ** -0.5
        dot_product = tf.matmul(q, k, transpose_b=True)  # [B, H, N, N]
        weights = K.softmax(dot_product)

        output = tf.matmul(weights, v)  # [B, H, N, V]

        # [B, H, N, V] -> [B, N, H, V]
        output_transpose = K.permute_dimensions(output, [0, 2, 1, 3])

        # [B, N, H, V] -> [B, N, H * V]
        new_memory = K.reshape(output_transpose, (batch_size, mem_slots, self.mem_size))

        return new_memory

    def _attention_mlp(self, memory):
        for i in range(self.attention_mlp_layers):
            memory = self._linear(memory, self.mlp_kernels[i], self.mlp_biases[i])
            if i < self.attention_mlp_layers - 1:
                memory = K.relu(memory)
        return memory

    def _create_gates(self, inputs, memory):
        """Create input and forget gates for this step using `inputs` and `memory`.
            Args:
              inputs: Tensor input.
              memory: The current state of memory.
            Returns:
              input_gate: A LSTM-like insert gate.
              forget_gate: A LSTM-like forget gate.
            """
        # We'll create the input and forget gates at once. Hence, calculate double
        # the gate size.
        memory = K.tanh(memory)
        inputs = K.batch_flatten(inputs)
        gate_inputs = self._linear(inputs, self.kernel_gi, self.bias_gi)
        gate_inputs = K.expand_dims(gate_inputs, axis=1)
        gate_memory = self._linear(memory, self.kernel_gm, self.bias_gm)
        gates = tf.split(gate_memory + gate_inputs, num_or_size_splits=2, axis=2)
        input_gate, forget_gate = gates

        input_gate = tf.sigmoid(input_gate + self.input_bias)
        forget_gate = tf.sigmoid(forget_gate + self.forget_bias)

        return input_gate, forget_gate

    def _layer_norm(self, x, offset, scale):
        in_shape = x.shape
        if len(in_shape) > 2:
            x_shape = [int(dim) for dim in x.shape]
            x = K.reshape(x, (x_shape[0] * x_shape[1], x_shape[2]))
        mean, var = tf.nn.moments(x, [1], keep_dims=True)
        x = tf.nn.batch_normalization(x, mean, var, offset, scale, K.epsilon())
        if len(in_shape) > 2:
            x = K.reshape(x, x_shape)
        return x

    def _linear(self, x, kernel, bias):
        in_shape = x.shape
        if len(in_shape) > 2:
            x_shape = [int(dim) for dim in x.shape]
            x = K.reshape(x, (x_shape[0] * x_shape[1], x_shape[2]))
        x = K.dot(x, kernel)
        x = K.bias_add(x, bias)
        if len(in_shape) > 2:
            x = K.reshape(x, (x_shape[0], x_shape[1], int(kernel.shape[1])))
        return x


class RelationalMemoryRNN(RNN):
    # pylint: disable=line-too-long
    """Constructs a `RelationalMemory` RNN.
    Args:
      mem_slots: The total number of memory slots to use.
      head_size: The size of an attention head.
      num_heads: The number of attention heads to use. Defaults to 1.
      forget_bias: Bias to use for the forget gate, assuming we are using
        some form of gating. Defaults to 1.
      input_bias: Bias to use for the input gate, assuming we are using
        some form of gating. Defaults to 0.
      gate_style: Whether to use per-element gating ('unit'),
        per-memory slot gating ('memory'), or no gating at all (None).
        Defaults to `unit`.
      attention_mlp_layers: Number of layers to use in the post-attention
        MLP. Defaults to 2.
      key_size: Size of vector to use for key & query vectors in the attention
        computation. Defaults to None, in which case we use `head_size`.
    """

    # pylint: enable=line-too-long

    def __init__(self, mem_slots, head_size, num_heads=1,
                 forget_bias=1.0, input_bias=0.0, gate_style='memory',
                 attention_mlp_layers=2, key_size=None,
                 return_sequences=False, return_state=False, go_backwards=False,
                 stateful=False, unroll=False, **kwargs):
        cell = RelationalMemoryCell(mem_slots, head_size, num_heads=num_heads,
                                    forget_bias=forget_bias, input_bias=input_bias, gate_style=gate_style,
                                    attention_mlp_layers=attention_mlp_layers, key_size=key_size)
        super(RelationalMemoryRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(RelationalMemoryRNN, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    def get_initial_state(self, inputs):
        batch_size = int(inputs.shape[0])
        mem_size = self.num_heads * self.head_size
        initial_state = tf.eye(self.mem_slots, batch_shape=[batch_size])

        # Pad the matrix with zeros.
        if mem_size > self.mem_slots:
            difference = mem_size - self.mem_slots
            pad = tf.zeros((batch_size, self.mem_slots, difference))
            initial_state = tf.concat([initial_state, pad], -1)
        # Truncation. Take the first `self._mem_size` components.
        elif mem_size < self.mem_slots:
            initial_state = initial_state[:, :, :mem_size]

        initial_state = tf.reshape(initial_state, (batch_size, self.units))
        return [initial_state]

    @property
    def mem_slots(self):
        return self.cell.mem_slots

    @property
    def head_size(self):
        return self.cell.head_size

    @property
    def num_heads(self):
        return self.cell.num_heads

    @property
    def units(self):
        return self.cell.mem_slots * self.cell.mem_size

    @property
    def forget_bias(self):
        return self.cell.forget_bias

    @property
    def input_bias(self):
        return self.cell.input_bias

    @property
    def gate_style(self):
        return self.cell.gate_style

    @property
    def attention_mlp_layers(self):
        return self.cell.attention_mlp_layers

    @property
    def key_size(self):
        return self.cell.key_size

    def get_config(self):
        config = {
            'mem_slots': self.mem_slots,
            'head_size': self.head_size,
            'num_heads': self.num_heads,
            'forget_bias': self.forget_bias,
            'input_bias': self.input_bias,
            'gate_style': self.gate_style,
            'attention_mlp_layers': self.attention_mlp_layers,
            'key_size': self.key_size
        }
        base_config = super(RelationalMemoryRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

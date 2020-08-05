import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Dot, add
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras import initializers, activations, regularizers, constraints
from tensorflow.keras.backend import bias_add
from tensorflow.keras.backend import dot


class ResidualLSTMCell(Layer):
    """
    The structure of the LSTM allows it to learn on problems with
    long term dependencies relatively easily. The "long term"
    memory is stored in a vector of memory cells c.
    Although many LSTM architectures differ in their connectivity
    structure and activation functions, all LSTM architectures have
    memory cells suitable for storing information for long periods
    of time. Here we implement the ResNet inspired Residual-LSTM.
    SELU is chosen as activation function of easier module due to its
    inherent parameters, 'scale' and 'alpha'. The values of `alpha`
    and `scale` are chosen so that the mean and variance of the inputs
    are preserved between two consecutive layers as long as the weights
    are initialized according to `lecun_normal` or 'Glorot normal'
    initialization and the number of inputs is "large enough"
    """
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 easier_activation='selu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ResidualLSTMCell, self).__init__(**kwargs)
        self.units = units,
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.easier_activation = activations.get(easier_activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.state_size = (self.units, self.units)
        self.output_size = self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units[0] * 5),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
                                      # trainable=True)
        self.recurrent_kernel = self.add_weight(shape=(self.units[0], self.units[0] * 5),
                                      name='recurrent_kernel',
                                      initializer=self.recurrent_initializer,
                                      regularizer=self.recurrent_regularizer,
                                      constraint=self.recurrent_constraint)
                                      # trainable=True)
        easier_kernel = self.add_weight(shape=(self.units[0], self.units[0]),
                                      name='easier_kernel',
                                      initializer=self.kernel_initializer)
                                      # trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units[0] * 6,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
                                        # trainable=True)
        else:
            self.bias = None

        self.kernel_i      = self.kernel[:, :self.units[0]]
        self.kernel_f      = self.kernel[:, self.units[0]: self.units[0] * 2]
        self.kernel_c      = self.kernel[:, self.units[0] * 2: self.units[0] * 3]
        self.kernel_o      = self.kernel[:, self.units[0] * 3: self.units[0] * 4]
        self.kernel_r      = self.kernel[:, self.units[0] * 4:]
        self.easier_kernel = easier_kernel

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units[0]]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units[0]: self.units[0] * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units[0] * 2: self.units[0] * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units[0] * 3: self.units[0] * 4]
        self.recurrent_kernel_r = self.recurrent_kernel[:, self.units[0] * 4:]

        if self.use_bias:
            self.bias_i      = self.bias[:self.units[0]]
            self.bias_f      = self.bias[self.units[0]: self.units[0] * 2]
            self.bias_c      = self.bias[self.units[0] * 2: self.units[0] * 3]
            self.bias_o      = self.bias[self.units[0] * 3: self.units[0] * 4]
            self.bias_r      = self.bias[self.units[0] * 4: self.units[0] * 5]
            self.bias_easier = self.bias[self.units[0] * 5:]
        else:
            self.bias_i      = None
            self.bias_f      = None
            self.bias_c      = None
            self.bias_o      = None
            self.bias_r      = None
            self.bias_easier = None

    def call(self, inputs, states):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs
        inputs_r = inputs

        x_i = dot(inputs_i, self.kernel_i)
        x_f = dot(inputs_f, self.kernel_f)
        x_c = dot(inputs_c, self.kernel_c)
        x_o = dot(inputs_o, self.kernel_o)
        x_r = dot(inputs_r, self.kernel_r)

        if self.use_bias:
            x_i = bias_add(x_i, self.bias_i)
            x_f = bias_add(x_f, self.bias_f)
            x_c = bias_add(x_c, self.bias_c)
            x_o = bias_add(x_o, self.bias_o)
            x_r = bias_add(x_r, self.bias_r)

        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
        h_tm1_r = h_tm1
        i = self.recurrent_activation(x_i + dot(h_tm1_i, self.recurrent_kernel_i))
        f = self.recurrent_activation(x_f + dot(h_tm1_f, self.recurrent_kernel_f))
        c = f * c_tm1 + i * self.activation(x_c + dot(h_tm1_c, self.recurrent_kernel_c))
        o = self.recurrent_activation(x_o + dot(h_tm1_o, self.recurrent_kernel_o))
        h = o * self.activation(c)
        h = self.easier_activation(dot(h, self.easier_kernel))
        identity = self.activation(x_r + dot(h_tm1_r, self.recurrent_kernel_r))
        h = add([h, identity])
        return h, [h, c]

    def get_config(self):
        config = {
            'units':
                self.units,
            'activation':
                activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint)
        }
        base_config = super(ResidualLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ResidualLSTM(RNN):
    """Residual Long Short-Term Memory layer - Nikhil 2020.
    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
      recurrent_activation: Activation function to use
        for the recurrent step.
        Default: hard sigmoid (`hard_sigmoid`).
        If you pass `None`, no activation is applied
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs..
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      return_sequences: Boolean. Whether to return the last output.
        in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
        in addition to the output.
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      unroll: Boolean (default False).
        If True, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.
      time_major: The shape format of the `inputs` and `outputs` tensors.
        If True, the inputs and outputs will be in shape
        `(timesteps, batch, ...)`, whereas in the False case, it will be
        `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
        efficient because it avoids transposes at the beginning and end of the
        RNN calculation. However, most TensorFlow data is batch-major, so by
        default this function accepts input and emits output in batch-major
        form.
    Call arguments:
      inputs: A 3D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
    """

    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        cell = ResidualLSTMCell(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True))
        super(ResidualLSTM, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.input_spec = [InputSpec(ndim=3)]

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(ResidualLSTM, self).call(
            inputs,
            mask=mask,
            training=training,
            initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    def get_config(self):
        config = {
            'units':
                self.units,
            'activation':
                activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
        }
        base_config = super(ResidualLSTM, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))


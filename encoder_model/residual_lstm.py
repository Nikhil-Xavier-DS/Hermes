import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer, RNN, GRU, Embedding
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras import initializers, activations, regularizers, constraints


class GRUEncoder(Model):
    """
    Create GRU Encoder block
    """
    def __init__(self, vocab_size,
                 embedding_dim,
                 enc_units,
                 batch_sz=1,
                 return_sequence=True,
                 return_state=True,
                 recurrent_initializer='glorot_uniform'):
        super(GRUEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.enc_units,
                       return_sequence=return_sequence,
                       return_state=return_state,
                       recurrent_initializer=recurrent_initializer)

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state


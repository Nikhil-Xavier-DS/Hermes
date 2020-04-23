import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer, Dense, RNN, GRU, LSTM, Embedding


class RNNEncoder(Model):
    """
    Create RNN Decoder block
    """

    def __init__(self, vocab_size,
                 embedding_dim,
                 dec_units,
                 batch_sz=1,
                 attention=None,
                 return_sequence=True,
                 return_state=True,
                 recurrent_initializer='glorot_uniform'):
        super(RNNDecoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.return_state = return_state
        self.return_sequence = return_sequence
        self.rnn = RNN(self.dec_units,
                       return_sequence=return_sequence,
                       return_state=return_state,
                       recurrent_initializer=recurrent_initializer)
        self.fc = Dense(vocab_size)
        self.attention = attention

    def call(self, x, hidden=None, encoder_output=None):
        context_vector, attention_weights = self.attention(hidden, encoder_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=1)
        output, state = self.rnn(inputs=x, initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights


class GRUDecoder(Model):
    """
    Create GRU Decoder block
    """

    def __init__(self, vocab_size,
                 embedding_dim,
                 dec_units,
                 batch_sz=1,
                 attention=None,
                 return_sequence=True,
                 return_state=True,
                 recurrent_initializer='glorot_uniform'):
        super(GRUDEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.return_state = return_state
        self.return_sequence = return_sequence
        self.gru = GRU(self.dec_units,
                       return_sequence=return_sequence,
                       return_state=return_state,
                       recurrent_initializer=recurrent_initializer)
        self.fc = Dense(vocab_size)
        self.attention = attention

    def call(self, x, hidden=None, encoder_output=None):
        context_vector, attention_weights = self.attention(hidden, encoder_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=1)
        output, state = self.gru(inputs=x, initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights


class LSTMDecoder(Model):
    """
    Create LSTM Decoder block
    """

    def __init__(self, vocab_size,
                 embedding_dim,
                 dec_units,
                 batch_sz=1,
                 attention=None,
                 return_sequence=True,
                 return_state=True,
                 recurrent_initializer='glorot_uniform'):
        super(GRUDEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.return_state = return_state
        self.return_sequence = return_sequence
        self.lstm = LSTM(self.dec_units,
                         return_sequence=return_sequence,
                         return_state=return_state,
                         recurrent_initializer=recurrent_initializer)
        self.fc = Dense(vocab_size)
        self.attention = attention

    def call(self, x, hidden_h=None, hidden_c=None,  encoder_output=None):
        context_vector, attention_weights = self.attention(hidden_h, encoder_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=1)
        output, h_state, c_state = self.lstm(inputs=x, initial_state=[hidden_h, hidden_c])
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, h_state, c_state, attention_weights


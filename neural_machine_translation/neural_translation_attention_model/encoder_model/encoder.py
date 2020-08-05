import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer, RNN, GRU, LSTM, Bidirectional, Embedding
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras import initializers, activations, regularizers, constraints


class RNNEncoder(Model):
    """
    Create RNN Encoder block
    """

    def __init__(self, vocab_size,
                 embedding_dim,
                 enc_units,
                 batch_sz=1,
                 return_sequences=True,
                 return_state=True,
                 recurrent_initializer='glorot_uniform'):
        super(RNNEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.rnn = RNN(self.enc_units,
                       return_sequences=return_sequences,
                       return_state=return_state,
                       recurrent_initializer=recurrent_initializer)

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

    def call(self, x, hidden=None):
        x = self.embedding(x)
        if self.return_state:
            if hidden is None:
                output, state = self.rnn(x)
            else:
                output, state = self.rnn(x, initial_state=hidden)
            return output, state
        else:
            if hidden is None:
                output = self.rnn(x)
            else:
                output = self.rnn(x, initial_state=hidden)
            return output


class GRUEncoder(Model):
    """
    Create GRU Encoder block
    """

    def __init__(self, vocab_size,
                 embedding_dim,
                 enc_units,
                 batch_sz=1,
                 return_sequences=True,
                 return_state=True,
                 recurrent_initializer='glorot_uniform'):
        super(GRUEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.gru = GRU(self.enc_units,
                       return_sequences=return_sequences,
                       return_state=return_state,
                       recurrent_initializer=recurrent_initializer)

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

    def call(self, x, hidden=None):
        x = self.embedding(x)
        if self.return_state:
            if hidden is None:
                output, state = self.gru(x)
            else:
                output, state = self.gru(x, initial_state=hidden)
            return output, state
        else:
            if hidden is None:
                output = self.gru(x)
            else:
                output = self.gru(x, initial_state=hidden)
            return output


class LSTMEncoder(Model):
    """
    Create LSTM Encoder block
    """

    def __init__(self, vocab_size,
                 embedding_dim,
                 enc_units,
                 batch_sz=1,
                 return_sequences=True,
                 return_state=True,
                 recurrent_initializer='glorot_uniform'):
        super(LSTMEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.lstm = LSTM(self.enc_units,
                         return_sequences=return_sequences,
                         return_state=return_state,
                         recurrent_initializer=recurrent_initializer)

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))

    def call(self, x, hidden_h=None, hidden_c=None):
        x = self.embedding(x)
        if self.return_state:
            if (hidden_c is None) or (hidden_h is None):
                output, h_state, c_state = self.lstm(x)
            else:
                output, h_state, c_state = self.lstm(x, initial_state=[hidden_h, hidden_c])
            return output, h_state, c_state
        else:
            if (hidden_c is None) or (hidden_h is None):
                output = self.lstm(x)
            else:
                output = self.lstm(x, initial_state=[hidden_h, hidden_c])
            return output


class BLSTMEncoder(Model):
    """
    Create Bidirectional LSTM Encoder block
    """

    def __init__(self, vocab_size,
                 embedding_dim,
                 enc_units,
                 batch_sz=1,
                 return_sequences=True,
                 return_state=True,
                 recurrent_initializer='glorot_uniform',
                 merge_mode='concat'):
        super(BLSTMEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.merge_mode = merge_mode
        self.blstm = Bidirectional(LSTM(self.enc_units,
                                        return_sequences=return_sequences,
                                        return_state=return_state,
                                        recurrent_initializer=recurrent_initializer),
                                   merge_mode=merge_mode)

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units)), \
               tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))

    def call(self, x, hidden_h_f=None, hidden_h_b=None, hidden_c_f=None, hidden_c_b=None):
        x = self.embedding(x)
        if self.return_state:
            if (hidden_h_f is None) or (hidden_c_f is None) or (hidden_h_b is None) or (hidden_c_b is None):
                output, h_state_f, c_state_f, h_state_b, c_state_b = self.blstm(x)
            else:
                output, h_state_f, c_state_f, h_state_b, c_state_b = self.blstm(x,
                                                                                initial_state=[hidden_h_f, hidden_c_f,
                                                                                               hidden_h_b, hidden_c_b])
            return output, h_state_f, c_state_f, h_state_b, c_state_b
        else:
            if (hidden_h_f is None) or (hidden_c_f is None) or (hidden_h_b is None) or (hidden_c_b is None):
                output = self.blstm(x)
            else:
                output = self.blstm(x, initial_state=[hidden_h_f, hidden_c_f, hidden_h_b, hidden_c_b])
            return output


class HybridBLSTMEncoder(Model):
    """
    Create Hybrid Bidirectional LSTM Encoder block to conveniently link with LSTM block
    """

    def __init__(self, vocab_size,
                 embedding_dim,
                 enc_units,
                 batch_sz=1,
                 return_sequences=True,
                 return_state=True,
                 recurrent_initializer='glorot_uniform',
                 merge_mode='concat'):
        super(HybridBLSTMEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.merge_mode = merge_mode
        self.blstm = Bidirectional(LSTM(self.enc_units,
                                        return_sequences=return_sequences,
                                        return_state=return_state,
                                        recurrent_initializer=recurrent_initializer),
                                   merge_mode=merge_mode)
        self.fc_h = tf.keras.layers.Dense(enc_units)
        self.fc_c = tf.keras.layers.Dense(enc_units)

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units)), \
               tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))

    def call(self, x, hidden_h_f=None, hidden_h_b=None, hidden_c_f=None, hidden_c_b=None):
        x = self.embedding(x)
        if self.return_state:
            if (hidden_h_f is None) or (hidden_c_f is None) or (hidden_h_b is None) or (hidden_c_b is None):
                output, h_state_f, c_state_f, h_state_b, c_state_b = self.blstm(x)
            else:
                output, h_state_f, c_state_f, h_state_b, c_state_b = self.blstm(x,
                                                                                initial_state=[hidden_h_f, hidden_c_f,
                                                                                               hidden_h_b, hidden_c_b])
            h_state = tf.concat([h_state_f, h_state_b], axis=1)
            c_state = tf.concat([c_state_f, c_state_b], axis=1)
            return output, h_state, c_state
        else:
            if (hidden_h_f is None) or (hidden_c_f is None) or (hidden_h_b is None) or (hidden_c_b is None):
                output = self.blstm(x)
            else:
                output = self.blstm(x, initial_state=[hidden_h_f, hidden_c_f, hidden_h_b, hidden_c_b])
            return output


class BGRUEncoder(Model):
    """
    Create Bidirectional GRU Encoder block
    """

    def __init__(self, vocab_size,
                 embedding_dim,
                 enc_units,
                 batch_sz=1,
                 return_sequences=True,
                 return_state=True,
                 recurrent_initializer='glorot_uniform',
                 merge_mode='concat'):
        super(BGRUEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.merge_mode = merge_mode
        self.bgru = Bidirectional(GRU(self.enc_units,
                                      return_sequences=return_sequences,
                                      return_state=return_state,
                                      recurrent_initializer=recurrent_initializer),
                                  merge_mode=merge_mode)

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))

    def call(self, x, hidden_f=None, hidden_b=None):
        x = self.embedding(x)
        if self.return_state:
            if (hidden_f is None) or (hidden_b is None):
                output, state_f, state_b = self.bgru(x)
            else:
                output, state_f, state_b = self.bgru(x, initial_state=[hidden_f, hidden_b])
            return output, state_f, state_b
        else:
            if (hidden_f is None) or (hidden_b is None):
                output = self.bgru(x)
            else:
                output = self.bgru(x, initial_state=[hidden_f, hidden_b])
            return output


class HybridBGRUEncoder(Model):
    """
    Create Hybrid Bidirectional GRU Encoder block to conveniently link with GRU block
    """

    def __init__(self, vocab_size,
                 embedding_dim,
                 enc_units,
                 batch_sz=1,
                 return_sequences=True,
                 return_state=True,
                 recurrent_initializer='glorot_uniform',
                 merge_mode='concat'):
        super(HybridBGRUEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.merge_mode = merge_mode
        self.bgru = Bidirectional(GRU(self.enc_units,
                                      return_sequences=return_sequences,
                                      return_state=return_state,
                                      recurrent_initializer=recurrent_initializer),
                                  merge_mode=merge_mode)
        self.fc = tf.keras.layers.Dense(enc_units)

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))

    def call(self, x, hidden_f=None, hidden_b=None):
        x = self.embedding(x)
        if self.return_state:
            if (hidden_f is None) or (hidden_b is None):
                output, state_f, state_b = self.bgru(x)
            else:
                output, state_f, state_b = self.bgru(x, initial_state=[hidden_f, hidden_b])
            state = tf.concat([state_f, state_b], axis=1)
            return output, state
        else:
            if (hidden_f is None) or (hidden_b is None):
                output = self.bgru(x)
            else:
                output = self.bgru(x, initial_state=[hidden_f, hidden_b])
            return output


class BRNNEncoder(Model):
    """
    Create Bidirectional RNN Encoder block
    """

    def __init__(self, vocab_size,
                 embedding_dim,
                 enc_units,
                 batch_sz=1,
                 return_sequences=True,
                 return_state=True,
                 recurrent_initializer='glorot_uniform',
                 merge_mode='concat'):
        super(BRNNEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.merge_mode = merge_mode
        self.brnn = Bidirectional(RNN(self.enc_units,
                                      return_sequences=return_sequences,
                                      return_state=return_state,
                                      recurrent_initializer=recurrent_initializer),
                                  merge_mode=merge_mode)

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))

    def call(self, x, hidden_f=None, hidden_b=None):
        x = self.embedding(x)
        if self.return_state:
            if (hidden_f is None) or (hidden_b is None):
                output, state_f, state_b = self.brnn(x)
            else:
                output, state_f, state_b = self.brnn(x, initial_state=[hidden_f, hidden_b])
            return output, state_f, state_b
        else:
            if (hidden_f is None) or (hidden_b is None):
                output = self.brnn(x)
            else:
                output = self.brnn(x, initial_state=[hidden_f, hidden_b])
            return output


class HybridBRNNEncoder(Model):
    """
    Create Hybrid Bidirectional RNN Encoder block to conveniently link with RNU block
    """

    def __init__(self, vocab_size,
                 embedding_dim,
                 enc_units,
                 batch_sz=1,
                 return_sequences=True,
                 return_state=True,
                 recurrent_initializer='glorot_uniform',
                 merge_mode='concat'):
        super(HybridBRNNEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.merge_mode = merge_mode
        self.brnn = Bidirectional(RNN(self.enc_units,
                                      return_sequences=return_sequences,
                                      return_state=return_state,
                                      recurrent_initializer=recurrent_initializer),
                                  merge_mode=merge_mode)
        self.fc = tf.keras.layers.Dense(enc_units)

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))

    def call(self, x, hidden_f=None, hidden_b=None):
        x = self.embedding(x)
        if self.return_state:
            if (hidden_f is None) or (hidden_b is None):
                output, state_f, state_b = self.brnn(x)
            else:
                output, state_f, state_b = self.brnn(x, initial_state=[hidden_f, hidden_b])
            state = tf.concat([state_f, state_b], axis=1)
            return output, state
        else:
            if (hidden_f is None) or (hidden_b is None):
                output = self.brnn(x)
            else:
                output = self.brnn(x, initial_state=[hidden_f, hidden_b])
            return output


if __name__ == '__main__':
    batch_sz = 2
    enc_units = 10
    sequence_length = 8
    encoder_test = GRUEncoder(vocab_size=100, embedding_dim=16, enc_units=enc_units, batch_sz=batch_sz)
    sample_hidden = encoder_test.initialize_hidden_state()
    sample_input = tf.ones([batch_sz, sequence_length])
    sample_output, sample_hidden = encoder_test(sample_input, sample_hidden)

    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

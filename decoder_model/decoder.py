import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer, Dense, RNN, GRU, LSTM, Embedding
from Hermes.attention_model.attention import BahdanauAttention


class RNNDecoder(Model):
    """
    Create RNN Decoder block
    """

    def __init__(self, vocab_size,
                 embedding_dim,
                 dec_units,
                 att_units=None,
                 batch_sz=1,
                 attention='Bahdanau',
                 return_sequences=True,
                 return_state=True,
                 recurrent_initializer='glorot_uniform'):
        super(RNNDecoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        if att_units is not None:
            self.att_units = att_units
        else:
            self.att_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.rnn = RNN(self.dec_units,
                       return_sequences=return_sequences,
                       return_state=return_state,
                       recurrent_initializer=recurrent_initializer)
        self.fc = Dense(vocab_size)
        if attention == 'Bahdanau':
            self.attention = BahdanauAttention(self.att_units)

    def call(self, x, hidden=None, encoder_output=None):
        context_vector, attention_weights = self.attention(hidden, encoder_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
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
                 att_units=None,
                 batch_sz=1,
                 attention='Bahdanau',
                 return_sequences=True,
                 return_state=True,
                 recurrent_initializer='glorot_uniform'):
        super(GRUDecoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        if att_units is not None:
            self.att_units = att_units
        else:
            self.att_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.gru = GRU(self.dec_units,
                       return_sequences=return_sequences,
                       return_state=return_state,
                       recurrent_initializer=recurrent_initializer)
        self.fc = Dense(vocab_size)
        if attention == 'Bahdanau':
            self.attention = BahdanauAttention(self.att_units)

    def call(self, x, hidden=None, encoder_output=None):
        context_vector, attention_weights = self.attention(hidden, encoder_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
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
                 att_units=None,
                 batch_sz=1,
                 attention='Bahdanau',
                 return_sequences=True,
                 return_state=True,
                 recurrent_initializer='glorot_uniform'):
        super(LSTMDecoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        if att_units is not None:
            self.att_units = att_units
        else:
            self.att_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.lstm = LSTM(self.dec_units,
                         return_sequences=return_sequences,
                         return_state=return_state,
                         recurrent_initializer=recurrent_initializer)
        self.fc = Dense(vocab_size)
        if attention == 'Bahdanau':
            self.attention = BahdanauAttention(self.att_units)

    def call(self, x, hidden_h=None, hidden_c=None, encoder_output=None):
        context_vector, attention_weights = self.attention(hidden_h, encoder_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, h_state, c_state = self.lstm(inputs=x, initial_state=[hidden_h, hidden_c])
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, h_state, c_state, attention_weights


if __name__ == '__main__':
    batch_sz = 2
    enc_units = 10
    dec_units = 5
    sequence_length = 8
    decoder_test = GRUDecoder(vocab_size=50, embedding_dim=16, dec_units=dec_units, batch_sz=batch_sz)
    sample_decoder_hidden = tf.ones([batch_sz, dec_units])
    sample_encoder_output = tf.ones([batch_sz, sequence_length, enc_units])
    sample_decoder_input = tf.zeros([batch_sz, 1])
    sample_decoder_output, sample_decoder_state, attention_weights = decoder_test(sample_decoder_input,
                                                                                  sample_decoder_hidden,
                                                                                  sample_encoder_output)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
    print('Decoder state shape: (batch size, units) {}'.format(sample_decoder_state.shape))
    print('Decode attention weights shape: (batch size, sequence_len, 1) {}'.format(attention_weights.shape))

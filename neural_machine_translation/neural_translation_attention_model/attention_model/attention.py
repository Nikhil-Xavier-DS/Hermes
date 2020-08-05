import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Dense


class BahdanauAttention(Layer):
    """
    Create Bahdanau Attention layer
    """

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.units = units
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        hidden_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights*values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


if __name__ == '__main__':

    batch_sz = 2
    enc_units = 10
    dec_units = 5
    att_units = 4
    sequence_length = 8
    attention_layer = BahdanauAttention(att_units)
    sample_decoder_hidden = tf.ones([batch_sz, dec_units])
    sample_encoder_output = tf.ones([batch_sz, sequence_length, enc_units])
    attention_result, attention_weights = attention_layer(sample_decoder_hidden, sample_encoder_output)
    print("Attention result shape: (batch size, att_units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
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

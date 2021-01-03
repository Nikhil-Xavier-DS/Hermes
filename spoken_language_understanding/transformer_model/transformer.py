# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A Transformer Encoder Model for Spoken Language Understanding"""
import tensorflow as tf
import time
import logging
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Layer
from tensorflow.keras.optimizers import Adam


class LayerNorm(Layer):
    def __init__(self, units=300):
        super().__init__()
        self.epsilon = 1e-6
        self.units = units
        self.scale = self.add_weight(shape=[self.units],
                                     initializer=tf.ones_initializer(),
                                     trainable=True)
        self.bias = self.add_weight(shape=[self.units],
                                    initializer=tf.zeros_initializer(),
                                    trainable=True)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axis=[-1], keepdims=True)
        norm_x = (inputs - mean) * tf.math.rsqrt(variance + self.epsilon)
        return norm_x * self.scale + self.bias


class EncoderBlock(Model):
    def __init__(self, SubModel, units=300, hidden_units=512, num_heads=8, multiplier=2, dropout_rate=0.2):
        super().__init__()
        self.layer_norm = LayerNorm(units)
        self.sub_model = SubModel(units, hidden_units, num_heads, multiplier, dropout_rate)
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        inputs, masks = inputs
        x = self.layer_norm(inputs)
        x = self.sub_model((x, masks), training=training)
        x = self.dropout(x, training=training)
        x += inputs
        return x


class PointwiseFFN(Model):
    def __init__(self, units=300, hidden_units=512, num_heads=8, multiplier=2, dropout_rate=0.2):
        super().__init__()
        self.dense_1 = Dense(multiplier * units, activation=tf.nn.elu)
        self.dropout = Dropout(dropout_rate)
        self.dense_2 = Dense(units)

    def call(self, inputs, training=True):
        x, masks = inputs
        return self.dense_2(self.dropout(self.dense_1(x), training=training))


class MultiheadSelfAttention(Model):
    def __init__(self, units=300, hidden_units=512, num_heads=8, multiplier=2, dropout_rate=0.2):
        super().__init__()
        self.qkv_linear = tf.keras.layers.Dense(3 * hidden_units)
        self.dropout = Dropout(dropout_rate)
        self.out_linear = Dense(units, activation=tf.nn.elu)
        self.num_heads = num_heads
        self.is_bidirectional = True

    def call(self, inputs, training):
        x, masks = inputs
        time_steps = tf.shape(x)[1]
        q_k_v = self.qkv_linear(x)
        q, k, v = tf.split(q_k_v, 3, axis=-1)

        if self.num_heads > 1:
            q = tf.concat(tf.split(q, self.num_heads, axis=-1), axis=0)
            k = tf.concat(tf.split(k, self.num_heads, axis=-1), axis=0)
            v = tf.concat(tf.split(v, self.num_heads, axis=-1), axis=0)

        align = tf.matmul(q, k, transpose_b=True) * tf.math.rsqrt(tf.cast(k.shape[-1], tf.float32))
        if (masks is not None) or (not self.is_bidirectional):
            paddings = tf.fill(tf.shape(align), float('-inf'))

        if masks is not None:
            c_masks = tf.tile(masks, [self.num_heads, 1])
            c_masks = tf.tile(tf.expand_dims(c_masks, 1), [1, time_steps, 1])
            align = tf.where(tf.equal(c_masks, 0), paddings, align)

        if not self.is_bidirectional:
            lower_tri = tf.linalg.LinearOperatorLowerTriangular(tf.ones(shape=(time_steps, time_steps))).to_dense()
            causal_masks = tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(align)[0], 1, 1])
            align = tf.where(tf.equal(causal_masks, 0), paddings, align)

        align = tf.nn.softmax(align)
        align = self.dropout(align, training=training)

        if masks is not None:
            q_masks = tf.tile(masks, [self.num_heads, 1])
            q_masks = tf.tile(tf.expand_dims(q_masks, 2), [1, 1, time_steps])
            align = align * tf.cast(q_masks, tf.float32)

        x = tf.matmul(align, v)

        if self.num_heads > 1:
            x = tf.concat(tf.split(x, self.num_heads, axis=0), axis=2)
        x = self.out_linear(x)
        return x


class TransformerModel(Model):
    EPOCHS = 1
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.INFO)

    def __init__(self, intent_size, slot_size, lr=1e-4, units=300, hidden_units=512, num_heads=8, multiplier=2,
                 dropout_rate=0.2, num_layers=6):
        super().__init__()
        self.embedding = tf.Variable(np.load('../data/embedding.npy'),
                                     dtype=tf.float32,
                                     trainable=False)
        self.inp_dropout_rate = Dropout(dropout_rate)
        self.num_layers = num_layers
        self.blocks = []
        for i in range(self.num_layers):
            self.blocks.append(EncoderBlock(
                MultiheadSelfAttention, units, hidden_units, num_heads, multiplier, dropout_rate))
            self.blocks.append(EncoderBlock(
                PointwiseFFN, units, hidden_units, num_heads, multiplier, dropout_rate))

        self.intent_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.fc_intent = tf.keras.layers.Dense(units, tf.nn.elu)
        self.out_linear_intent = Dense(intent_size)
        self.out_linear_slot = Dense(slot_size)
        self.optimizer = Adam(lr)
        self.slots_accuracy = tf.keras.metrics.Accuracy()
        self.intent_accuracy = tf.keras.metrics.Accuracy()
        self.decay_lr = tf.optimizers.schedules.ExponentialDecay(lr, 1000, 0.95)
        self.logger = logging.getLogger('tensorflow')
        self.logger.setLevel(logging.INFO)

    @staticmethod
    def get_timing_signal_1d(time_steps, channels):
        start_id = 0
        min_timescale = 1.0
        max_timescale = 1e4
        position = tf.cast(tf.range(time_steps) + start_id, tf.float32)
        num_timescales = channels // 2
        log_timescale_increment = (
                tf.math.log(float(max_timescale) / float(min_timescale))
                / tf.maximum(tf.cast(num_timescales, tf.float32) - 1, 1))
        inv_timescales = min_timescale * tf.exp(tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.compat.v1.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, time_steps, channels])
        return signal

    def call(self, inputs, training):
        if inputs.dtype != tf.int32:
            inputs = tf.cast(inputs, tf.int32)
        masks = tf.sign(inputs)
        x = tf.nn.embedding_lookup(self.embedding, inputs)
        time_steps = tf.shape(x)[1]
        x += self.get_timing_signal_1d(time_steps, self.units)
        x = self.input_dropout(x, training=training)

        for block in self.blocks:
            x = block((x, masks), training=training)

        x_intent = tf.reduce_max(x, 1)
        x_intent = self.intent_dropout(x_intent, training=training)
        x_intent = self.out_linear_intent(self.fc_intent(x_intent))
        x_slot = self.out_linear_slot(x)
        return x_intent, x_slot

    def fit(self, data, epochs=EPOCHS):
        t0 = time.time()
        step = 0
        epoch = 1
        while epoch <= epochs:
            for words, (intent, slots) in data:
                with tf.GradientTape() as tape:
                    y_intent, y_slots = self(words, training=True)
                    loss_intent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=intent,
                                                                                                logits=y_intent,
                                                                                                label_smoothing=.2, ))
                    weights = tf.cast(tf.sign(slots), tf.float32)
                    padding = tf.constant(1e-2, tf.float32, weights.shape)
                    weights = tf.where(tf.equal(weights, 0.), padding, weights)
                    weights = tf.cast(weights, tf.float32)
                    loss_slots = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=intent,
                                                                                               logits=y_intent,
                                                                                               weights=weights,
                                                                                               label_smoothing=.2, ))
                    loss = loss_intent + loss_slots
                self.optimizer.lr.assign(self.decay_lr(step))
                grads = tape.gradient(loss, self.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 5.)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                if step % 100 == 0:
                    self.logger.info("Step {} | Loss: {:.4f} | Spent: {:.1f} secs | LR: {:.6f}".format(
                        step, loss.numpy().item(), time.time() - t0, self.optimizer.lr.numpy().item()))
                    t0 = time.time()
                step += 1
            epoch += 1
        return True

    def evaluate(self, data):
        self.slot_accuracy.reset_states()
        self.intent_accuracy.reset_states()
        for words, (intent, slots) in data:
            y_intent, y_slots = self(words, training=False)
            y_intent = tf.argmax(y_intent, -1)
            y_slots = tf.argmax(y_slots, -1)
            self.intent_accuracy.update_state(y_true=intent, y_pred=y_intent)
            self.slots_accuracy.update_state(y_true=slots, y_pred=y_slots)

        intent_accuracy = self.intent_accuracy.result().numpy()
        slots_accuracy = self.slots_accuracy.result().numpy()
        self.logger.info("Evaluation Accuracy (intent): {:.3f}".format(intent_accuracy))
        self.logger.info("Evaluation Accuracy (slots): {:.3f}".format(slots_accuracy))

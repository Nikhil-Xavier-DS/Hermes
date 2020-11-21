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

"""Parallel LSTM with Attention model with char and word embedding for text classification."""

import tensorflow as tf
import time
import logging
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Embedding, LSTM, Bidirectional
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.optimizers import Adam


class KernelAttentivePooling(Model):
    def __init__(self, dropout_rate=0.2, kernel_size=5):
        super().__init__()
        self.dropout = Dropout(dropout_rate)
        self.kernel = Conv1D(filters=1,
                             kernel_size=kernel_size,
                             padding='same',
                             activation='tanh',
                             use_bias=True)

    def call(self, inputs, training=False):
        x, masks = inputs
        x = self.dropout(x, training=training)
        x = self.kernel(x)
        align = tf.squeeze(x, -1)
        padding = tf.fill(tf.shape(align), float('-inf'))
        align = tf.where(tf.equal(masks, 0), padding, align)
        align = tf.nn.softmax(align)
        align = tf.expand_dims(align, -1)
        return tf.squeeze(tf.matmul(x, align, transpose_a=True), axis=-1)


class ParallelLSTMTextClassifier(Model):
    EPOCHS = 1
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.INFO)

    def __init__(self, lstm_units=100, char_vocab=26, char_embed_size=100, cnn_filters=300,
                 cnn_kernel_size=5, dropout_rate=0.2, max_char_len=10, lr=1e-4):
        super().__init__()
        self.char_embedding = Embedding(char_vocab + 1, char_embed_size)
        self.word_embedding = tf.Variable(np.load('../data/embedding.npy'),
                                          dtype=tf.float32,
                                          name='pretrained_embedding',
                                          trainable=False, )
        self.char_cnn = Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size,
                               activation='elu', padding='same')
        self.embed_drop = Dropout(dropout_rate)
        self.embed_fc = Dense(cnn_filters, 'elu', name='embed_fc')
        self.word_cnn = Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size,
                               activation='elu', padding='same')
        self.word_drop = Dropout(dropout_rate)
        self.max_char_len = max_char_len
        self.char_embed_size = char_embed_size
        self.cnn_filters = cnn_filters
        self.attentive_pooling = KernelAttentivePooling(dropout_rate)
        self.dropout_l1 = Dropout(dropout_rate)
        self.dropout_l2 = Dropout(dropout_rate)
        self.blstm_l1 = Bidirectional(LSTM(lstm_units, return_sequences=True))
        self.blstm_l2 = Bidirectional(LSTM(lstm_units, return_sequences=True))
        self.dropout_op = Dropout(dropout_rate)
        self.units = 2 * lstm_units
        self.fc = Dense(units=self.units, activation='elu')
        self.out_linear = Dense(2)
        self.optimizer = Adam(lr)
        self.accuracy = Accuracy()
        self.decay_lr = tf.optimizers.schedules.ExponentialDecay(lr, 1000, 0.95)
        self.logger = logging.getLogger('tensorflow')
        self.logger.setLevel(logging.INFO)

    def call(self, inputs, training=False):
        words, chars = inputs
        if words.dtype != tf.int32:
            words = tf.cast(words, tf.int32)
        masks = tf.sign(words)
        batch_size = tf.shape(words)[0]
        word_len = tf.shape(words)[1]
        chars = self.char_embedding(chars)
        chars = tf.reshape(chars, (batch_size * word_len, self.max_char_len, self.char_embed_size))
        chars = self.char_cnn(chars)
        chars = tf.reduce_max(chars, 1)
        chars = tf.reshape(chars, (batch_size, word_len, self.cnn_filters))
        words = tf.nn.embedding_lookup(self.word_embedding, words)
        x = tf.concat((words, chars), axis=-1)

        x = tf.reshape(x, (batch_size * 10 * 10, 10, self.embedding.shape[-1]))
        x = self.dropout_l1(x, training=training)
        x = self.blstm_l1(x)
        x = tf.reduce_max(x, 1)

        x = tf.reshape(x, (batch_size * 10, 10, self.units))
        x = self.dropout_l2(x, training=training)
        x = self.blstm_l2(x)
        x = tf.reduce_max(x, 1)

        x = tf.reshape(x, (batch_size, 10, self.units))
        masks = tf.reshape(tf.sign(tf.reduce_sum(masks, 1)), (self.batch_size, 10))
        x = self.attentive_pooling((x, masks), training=training)
        x = self.dropout_op(x, training=training)
        x = self.fc(x)
        x = self.out_linear(x)
        return x

    def fit(self, data, epochs=EPOCHS):
        t0 = time.time()
        step = 0
        epoch = 1
        while epoch <= epochs:
            for texts, labels in data:
                with tf.GradientTape() as tape:
                    logits = self.call(texts, training=True)
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                                         logits=logits,
                                                                                         label_smoothing=.2, ))
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
        self.accuracy.reset_states()
        for texts, labels in data:
            logits = self.call(texts, training=False)
            y_pred = tf.argmax(logits, axis=-1)
            self.accuracy.update_state(y_true=labels, y_pred=y_pred)

        accuracy = self.accuracy.result().numpy()
        self.logger.info("Evaluation Accuracy: {:.3f}".format(accuracy))
        self.logger.info("Accuracy: {:.3f}".format(accuracy))


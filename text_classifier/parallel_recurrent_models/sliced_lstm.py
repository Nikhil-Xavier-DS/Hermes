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

"""Parallel LSTM (Sliced) model for text classification."""

import tensorflow as tf
import time
import logging
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from tensorflow.optimizers.schedules import ExponentialDecay


class ParallelLSTMTextClassifier(Model):
    EPOCHS = 1
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.INFO)

    def __init__(self, lstm_units=100, lr=1e-4, dropout_rate=0.2):
        super().__init__()
        self.embedding = tf.Variable(np.load('../data/embedding.npy'),
                                     dtype=tf.float32,
                                     name='pretrained_embedding',
                                     trainable=False)
        self.dropout_l1 = Dropout(dropout_rate)
        self.dropout_l2 = Dropout(dropout_rate)
        self.dropout_l3 = Dropout(dropout_rate)
        self.blstm_l1 = Bidirectional(LSTM(lstm_units, return_sequences=True))
        self.blstm_l2 = Bidirectional(LSTM(lstm_units, return_sequences=True))
        self.blstm_l3 = Bidirectional(LSTM(lstm_units, return_sequences=True))
        self.dropout_op = Dropout(dropout_rate)
        self.units = 2 * lstm_units
        self.fc = Dense(units=self.units, activation='elu')
        self.out_linear = Dense(2)
        self.optimizer = Adam(lr)
        self.decay_lr = ExponentialDecay(lr, 1000, 0.90)
        self.accuracy = Accuracy()
        self.logger = logging.getLogger('tensorflow')
        self.logger.setLevel(logging.INFO)

    def call(self, inputs, training=False):
        if inputs.dtype != tf.int32:
            inputs = tf.cast(inputs, tf.int32)
        batch_size = tf.shape(inputs)[0]
        x = tf.nn.embedding_lookup(self.embedding, inputs)
        x = tf.reshape(x, (batch_size * 10 * 10, 10, self.embedding.shape[-1]))
        x = self.dropout_l1(x, training=training)
        x = self.blstm_l1(x)
        x = tf.reduce_max(x, 1)

        x = tf.reshape(x, (batch_size * 10, 10, self.units))
        x = self.dropout_l2(x, training=training)
        x = self.blstm_l2(x)
        x = tf.reduce_max(x, 1)

        x = tf.reshape(x, (batch_size, 10, self.units))
        x = self.dropout_l3(x, training=training)
        x = self.blstm_l3(x)
        x = tf.reduce_max(x, 1)

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

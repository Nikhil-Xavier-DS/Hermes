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

"""Text Matching as Image Recognition using Match Pyramid for Natural Language Inference"""

import tensorflow as tf
import time
import logging
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam


class FCBlock(Model):
    def __init__(self, dropout_rate=0.2, units=300):
        super().__init__()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.fc1 = tf.keras.layers.Dense(units, tf.nn.elu)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.fc2 = tf.keras.layers.Dense(units, tf.nn.elu)

    def call(self, inputs, training=False, **args):
        super().call(**args)
        x = inputs
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        return x


class MatchPyramidModel(Model):
    EPOCHS = 1
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.INFO)

    def __init__(self, lr=1e-4, dropout_rate=0.2, units=300, max_len1=16, max_len2=12):
        super().__init__()
        self.embedding = tf.Variable(np.load('../data/embedding.npy'),
                                     dtype=tf.float32,
                                     name='pretrained_embedding',
                                     trainable=False)
        self.inp_dropout = Dropout(rate=dropout_rate)
        self.encoder = Bidirectional(LSTM(units=units, return_sequences=True))
        self.conv_1 = Conv2D(filters=32, kernel_size=7, activation=tf.nn.elu, padding='same')
        self.conv_2 = Conv2D(filters=64, kernel_size=5, activation=tf.nn.elu, padding='same')
        self.conv_3 = Conv2D(filters=128, kernel_size=3, activation=tf.nn.elu, padding='same')
        self.W_0 = Dense(2*units)
        self.W_1_1 = Dense(units)
        self.W_1_2 = Dense(units)
        self.v_1 = Dense(1)
        self.W_2 = Dense(units)
        self.v_2 = Dense(1)
        self.W_3 = Dense(units)
        self.v_3 = Dense(1)
        self.flatten = Flatten()
        self.out_hidden = FCBlock(dropout_rate, units)
        self.out_linear = Dense(3)
        self.max_len1 = max_len1
        self.max_len2 = max_len2
        self.optimizer = Adam(lr)
        self.accuracy = tf.keras.metrics.Accuracy()
        self.decay_lr = tf.optimizers.schedules.ExponentialDecay(lr, 1000, 0.95)
        self.logger = logging.getLogger('tensorflow')
        self.logger.setLevel(logging.INFO)

    def call(self, inputs, training=False, **args):
        super().call(**args)
        x1, x2 = inputs

        if x1.dtype != tf.int32:
            x1 = tf.cast(x1, tf.int32)
        if x2.dtype != tf.int32:
            x2 = tf.cast(x2, tf.int32)

        batch_sz = tf.shape(x1)[0]
        len1 = x1.shape[1]
        len2 = x2.shape[1]
        stride1 = len1 // self.max_len1
        stride2 = len2 // self.max_len2

        if len1 // stride1 != self.max_len1:
            remainder = (stride1+1)*self.max_len1 - len1
            zeros = tf.zeros([batch_sz, remainder], tf.int32)
            x1 = tf.concat([x1, zeros], 1)
            len1 = x1.shape[1]
            stride1 = len1 // self.max_len1

        if len2 // stride2 != self.max_len2:
            remainder = (stride1 + 1) * self.max_len1 - len1
            zeros = tf.zeros([batch_sz, remainder], tf.int32)
            x2 = tf.concat([x2, zeros], 1)
            len2 = x2.shape[1]
            stride2 = len2 // self.max_len2

        mask1 = tf.sign(x1)
        mask2 = tf.sign(x2)
        x1 = tf.nn.embedding_lookup(self.embedding, x1)
        x2 = tf.nn.embedding_lookup(self.embedding, x2)
        x1 = self.inp_dropout(x1, training=training)
        x2 = self.inp_dropout(x2, training=training)
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x = []

        # feature map 1
        a = tf.matmul(x1, self.W_0(x2), transpose_b=True)
        x.append(tf.expand_dims(a, -1))

        # feature map 2
        a1 = tf.expand_dims(self.W_1_1(x1), 2)
        a2 = tf.expand_dims(self.W_1_2(x2), 1)
        x.append(self.v_1(tf.tanh(a1 + a2)))

        # feature map 3
        a1 = tf.expand_dims(x1, 2)
        a2 = tf.expand_dims(x2, 1)
        x.append(self.v_2(tf.tanh(self.W_2(tf.abs(a1 - a2)))))

        # feature map 4
        a1 = tf.expand_dims(x1, 2)
        a2 = tf.expand_dims(x2, 1)
        x.append(self.v_3(tf.tanh(self.W_3(a1 * a2))))

        x = tf.concat(x, -1)
        x = self.conv_1(x)
        x = tf.nn.max_pool(input=x, ksize=[1, stride1, stride2, 1], stride=[1, stride1, stride2, 1], padding='VALID')
        x = self.conv_2(x)
        x = tf.nn.max_pool(input=x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='VALID')
        x = self.conv_3(x)
        x = tf.nn.max_pool(input=x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='VALID')

        x = self.flatten(x)
        x = self.out_hidden(x, training=training)
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

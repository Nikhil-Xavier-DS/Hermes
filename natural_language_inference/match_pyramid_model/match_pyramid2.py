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

"""A Decomposable Attention Model for Natural Language Inference"""

import tensorflow as tf
import time
import logging
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
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

    def __init__(self, lr=1e-4, dropout_rate=0.2, units=300):
        super().__init__()
        self.embedding = tf.Variable(np.load('../data/embedding.npy'),
                                     dtype=tf.float32,
                                     name='pretrained_embedding',
                                     trainable=False)
        self.attentive_pooling = KernelAttentivePooling(dropout_rate)
        self.attend_transform = FCBlock(dropout_rate, units)
        self.compare_transform = FCBlock(dropout_rate, units)
        self.aggregate_transform = FCBlock(dropout_rate, units)
        self.soft_align_attention = SoftAlignAttention()
        self.out_linear = Dense(3)
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
        mask1 = tf.sign(x1)
        mask2 = tf.sign(x2)
        x1 = tf.nn.embedding_lookup(self.embedding, x1)
        x2 = tf.nn.embedding_lookup(self.embedding, x2)
        x1_coef, x2_coef = self.attend(x1, x2, mask1, mask2, training)
        x1 = self.compare(x1, x1_coef, training)
        x2 = self.compare(x2, x2_coef, training)
        x = self.aggregate(x1, x2, mask1, mask2, training)
        x = self.out_linear(x)
        return x

    def attend(self, x1, x2, mask1, mask2, training):
        x1 = self.attend_transform(x1, training=training)
        x2 = self.attend_transform(x2, training=training)
        return self.soft_align_attention(x1, x2, mask1, mask2)

    def compare(self, x, x_coef, training):
        def func(x, x_coef):
            return tf.concat((x,
                              x_coef,
                              (x - x_coef)), -1)

        x = func(x, x_coef)
        x = self.compare_transform(x, training=training)
        return x

    def aggregate(self, x1, x2, mask1, mask2, training):
        features = [tf.reduce_max(x1, axis=1),
                    tf.reduce_max(x2, axis=1),
                    self.attentive_pooling((x1, mask1), training),
                    self.attentive_pooling((x2, mask2), training)]
        x = tf.concat(features, axis=-1)
        x = self.aggregate_transform(x, training)
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

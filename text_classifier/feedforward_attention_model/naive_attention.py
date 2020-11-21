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

"""Feedforward Attention model for text classification."""

import tensorflow as tf
import time
import logging
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


class KernelAttentivePooling(Model):
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout = Dropout(dropout_rate)
        self.kernel = Dense(units=1,
                            activation="tanh",
                            use_bias=True)

    def call(self, inputs, training=False):
        x, masks = inputs
        x1 = self.dropout(x, training=training)
        x1 = self.kernel(x1)
        align = tf.squeeze(x1, -1)
        padding = tf.fill(tf.shape(align), float('-inf'))
        align = tf.where(tf.equal(masks, 0), padding, align)
        align = tf.nn.softmax(align)
        align = tf.expand_dims(align, -1)
        return tf.squeeze(tf.matmul(x, align, transpose_a=True), axis=-1)


class FeedForwardAttentionTextClassifier(Model):
    EPOCHS = 1
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.INFO)

    def __init__(self, lr=1e-4, dropout_rate=0.2):
        super().__init__()
        self.embedding = tf.Variable(np.load('../data/embedding.npy'),
                                     dtype=tf.float32,
                                     name='pretrained_embedding',
                                     trainable=False)
        self.attentive_pooling = KernelAttentivePooling(dropout_rate)
        self.out_linear = Dense(2)
        self.optimizer = Adam(lr)
        self.accuracy = tf.keras.metrics.Accuracy()
        self.decay_lr = tf.optimizers.schedules.ExponentialDecay(lr, 1000, 0.95)
        self.logger = logging.getLogger('tensorflow')
        self.logger.setLevel(logging.INFO)

    def call(self, inputs, training=False):
        if inputs.dtype != tf.int32:
            inputs = tf.cast(inputs, tf.int32)
        masks = tf.sign(inputs)
        x = tf.nn.embedding_lookup(self.embedding, inputs)
        x = self.attentive_pooling((x, masks), training=training)
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
                                                                                         label_smoothing=.2,))
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

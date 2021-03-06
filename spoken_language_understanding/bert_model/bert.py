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

"""A BERT (Bidirectional Encoder Representations from Transformers) Model for Spoken Language Understanding"""
import tensorflow as tf
import time
import logging
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from transformers import TFBertModel


class BERTModel(Model):
    EPOCHS = 1
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.INFO)

    def __init__(self, intent_size, slot_size, lr=1e-4, dropout_rate=0.2, units=300):
        super().__init__()
        self.bert = TFBertModel.from_pretrained('bert-base-uncased',
                                                trainable=True)
        self.inp_dropout = Dropout(dropout_rate)
        self.intent_dropout = Dropout(dropout_rate)
        self.fc_intent = Dense(units, activation='relu')
        self.trans_params = self.add_weight(shape=(slot_size, slot_size))
        self.out_linear_intent = Dense(intent_size)
        self.out_linear_slot = Dense(slot_size)
        self.optimizer = Adam(lr)
        self.slots_accuracy = tf.keras.metrics.Accuracy()
        self.intent_accuracy = tf.keras.metrics.Accuracy()
        self.decay_lr = tf.optimizers.schedules.ExponentialDecay(lr, 1000, 0.95)
        self.logger = logging.getLogger('tensorflow')
        self.logger.setLevel(logging.INFO)

    def call(self, bert_inps, training=False):
        bert_inps = [tf.cast(inp, tf.int32) for inp in bert_inps]
        x = self.bert(bert_inps, training=training)[0]
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
            for words, segs, (intent, slots) in data:
                with tf.GradientTape() as tape:
                    y_intent, y_slots = self([words, tf.sign(words), segs], training=True)
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

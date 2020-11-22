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

"""RoBERTa (Robustly Optimized BERT Pretraining Approach) model for text classification."""

import tensorflow as tf
import time
import logging

from tensorflow.keras import Model
from transformers import TFRobertaModel


class RobertClassifier(Model):
    EPOCHS = 1
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.INFO)

    def __init__(self, dropout=0.1):
        super().__init__()
        self.roberta = TFRobertaModel.from_pretrained('bert-base-uncased',
                                                trainable=True)
        self.drop = tf.keras.layers.Dropout(dropout)
        self.fc = tf.keras.layers.Dense(300, tf.nn.silu)
        self.out = tf.keras.layers.Dense(2)

    def call(self, roberta_inp, training):
        roberta_inp = [tf.cast(inp, tf.int32) for inp in roberta_inp]
        x = self.roberta(roberta_inp, training=training)[1]
        x = self.drop(x, training=training)
        x = self.fc(x)
        x = self.drop(x, training=training)
        x = self.out(x)
        return x

    def fit(self, data, epochs=EPOCHS):
        t0 = time.time()
        step = 0
        epoch = 1
        while epoch <= epochs:
            for texts, segs, labels in data:
                with tf.GradientTape() as tape:
                    logits = self.call([texts, tf.sign(texts), segs], training=True)
                    loss = tf.reduce_mean(self.compiled_loss(labels=tf.one_hot(labels, 2),
                                                             logits=logits))
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
        for texts, segs, labels in data:
            logits = self.call([texts, tf.sign(texts), segs], training=False)
            y_pred = tf.argmax(logits, axis=-1)
            self.accuracy.update_state(y_true=labels, y_pred=y_pred)
        accuracy = self.accuracy.result().numpy()
        self.logger.info("Evaluation Accuracy: {:.3f}".format(accuracy))
        self.logger.info("Accuracy: {:.3f}".format(accuracy))

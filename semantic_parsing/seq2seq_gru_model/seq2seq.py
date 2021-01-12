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

"""A GRU based Sequence to Sequence Model for Semantic Parsing"""

import time
import logging
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import Model
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, GRU, GRUCell, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.seq2seq import AttentionWrapper, BahdanauAttention, BasicDecoder, BeamSearchDecoder


class Encoder(Model):
    def __init__(self, units=300, dropout_rate=0.2):
        super().__init__()
        self.dropout = Dropout(dropout_rate)
        self.bgru = Bidirectional(GRU(units,
                                      return_state=True,
                                      return_sequences=True,
                                      zero_output_for_mask=True))
        self.fc = tf.keras.layers.Dense(units, 'relu')

    def call(self, inputs, mask, training):
        if mask.dtype != tf.bool:
            mask = tf.cast(mask, tf.bool)
        x = self.dropout(inputs, training=training)
        output, forward_state, backward_state = self.bgru(x, mask=mask)
        concat_state = tf.concat((forward_state, backward_state), axis=-1)
        state = self.fc(concat_state)
        return output, state


class ProjectedLayer(Layer):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def call(self, inputs):
        x = tf.matmul(inputs, self.embedding, transpose_b=True)
        return x


class Seq2SeqModel(Model):
    EPOCHS = 1
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.INFO)

    def __init__(self, lr=1e-4, dropout_rate=0.2, units=300, beam_width=12):
        super().__init__()
        self.embedding = tf.Variable(np.load('../data/embedding.npy'),
                                     dtype=tf.float32,
                                     name='pretrained_embedding',
                                     trainable=False)
        self.encoder = Encoder(units=units, dropout_rate=dropout_rate)
        self.attention_mechanism = BahdanauAttention(units=units)
        self.decoder_cell = AttentionWrapper(
            GRUCell(units),
            self.attention_mechanism,
            attention_layer_size=units)
        self.projected_layer = ProjectedLayer(self.embed.embedding)
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        self.decoder = BasicDecoder(
            self.decoder_cell,
            self.sampler,
            output_layer=self.projected_layer)
        self.beam_search = BeamSearchDecoder(
            self.decoder_cell,
            beam_width=beam_width,
            embedding_fn=lambda x: tf.nn.embedding_lookup(self.embedding, x),
            output_layer=self.projected_layer)
        self.optimizer = Adam(lr)
        self.accuracy = tf.keras.metrics.Accuracy()
        self.mean = tf.keras.metrics.Mean()
        self.decay_lr = tf.optimizers.schedules.ExponentialDecay(lr, 1000, 0.95)
        self.logger = logging.getLogger('tensorflow')
        self.logger.setLevel(logging.INFO)

    def call(self, inputs, training=True):
        if training:
            source, target_in = inputs
        else:
            source = inputs

        batch_size = tf.shape(source)[0]

        if source.dtype != tf.int32:
            source = tf.cast(source, tf.int32)

        source_embedded = tf.nn.embedding_lookup(self.embedding, source)
        mask = tf.sign(source)
        encoder_output, encoder_state = self.encoder(source_embedded, mask=mask, training=training)

        if training:
            target_in_embedded = tf.nn.embedding_lookup(self.embedding, target_in)
            self.attention_mechanism([encoder_output, tf.math.count_nonzero(source, 1)], setup_memory=True)
            decoder_initial_state = self.decoder_cell.get_initial_state(batch_size=batch_size,
                                                                        dtype=tf.float32)
            decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
            decoder_output, _, _ = self.decoder(
                inputs=target_in_embedded,
                initial_state=decoder_initial_state,
                sequence_length=tf.math.count_nonzero(target_in, 1, dtype=tf.int32))
            logits = decoder_output.rnn_output
        else:
            encoder_output = tfa.seq2seq.tile_batch(encoder_output, self.beam_width)
            encoder_seq_len = tfa.seq2seq.tile_batch(tf.math.count_nonzero(source, 1), self.beam_width)
            encoder_state = tfa.seq2seq.tile_batch(encoder_state, self.beam_width)
            self.attention_mechanism([encoder_output, encoder_seq_len], setup_memory=True)
            decoder_initial_state = self.decoder_cell.get_initial_state(batch_size=batch_size * self.beam_width,
                                                                        dtype=tf.float32)
            decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)

            decoder_output, _, _ = self.beam_search(
                None,
                start_tokens=tf.tile(tf.constant([1], tf.int32), [batch_size]),
                end_token=2,
                initial_state=decoder_initial_state)

            logits = decoder_output.predicted_ids[:, :, 0]

        return logits

    def fit(self, data, epochs=EPOCHS, vocab_size=9000):
        t0 = time.time()
        step = 0
        epoch = 1
        while epoch <= epochs:
            for (source, target_in, target_out) in data:
                with tf.GradientTape() as tape:
                    logits = self((source, target_in), training=True)
                    loss = tf.compat.v1.losses.softmax_cross_entropy(
                        onehot_labels=tf.one_hot(target_out, (9000+1)),
                        logits=logits,
                        weights=tf.cast(tf.sign(target_out), tf.float32),
                        label_smoothing=.2)
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
        self.mean.reset_states()

        for (source, target_in, target_out) in data:
            logits = self((source, target_in), training=False)

            for prediction, target in zip(logits.numpy(), target_out.numpy()):
                matched = np.array_equal(prediction, target)
                self.mean.update_state(int(matched))

        mean = self.mean.result().numpy()
        self.logger.info("Evaluation Mean: {:.3f}".format(mean))
        return True
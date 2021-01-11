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
    def __init__(self, units=300, beam_width=12, dropout_rate=0.2):
        super().__init__()
        self.embedding = tf.Variable(np.load('../data/embedding.npy'),
                                     dtype=tf.float32,
                                     name='pretrained_embedding',
                                     trainable=False)
        self.encoder = Encoder(units=units, dropout_rate=dropout_rate)
        self.dropout = Dropout(dropout_rate=dropout_rate)
        self.attention_mechanism = BahdanauAttention(units=units)
        self.decoder_cell = AttentionWrapper(
            GRUCell(units),
            self.attention_mechanism,
            attention_layer_size=units)
        self.projected_layer = ProjectedLayer(self.embed.embedding)
        self.teach_forcing = BasicDecoder(
            self.decoder_cell,
            tfa.seq2seq.sampler.TrainingSampler(),
            output_layer=self.proj_layer)
        self.beam_search = BeamSearchDecoder(
            self.decoder_cell,
            beam_width=beam_width,
            embedding_fn=lambda x: tf.nn.embedding_lookup(self.embedding, x),
            output_layer=self.projected_layer,
            maximum_iterations=80, )

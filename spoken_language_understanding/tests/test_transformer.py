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

"""Test case for Transformer Model for Spoken Language Understanding"""

import tensorflow as tf
import os

from Hermes.spoken_language_understanding.transformer_model.transformer import TransformerModel
from Hermes.spoken_language_understanding.dataset.loader import dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("TensorFlow Version", tf.__version__)
print('GPU Enabled:', tf.test.is_gpu_available())

params = {
    'train_path': '../data/atis_data/atis.train.w-intent.iob',
    'test_path': '../data/atis_data/atis.test.w-intent.iob',
    'word_path': '../data/word.txt',
    'intent_path': '../data/intent.txt',
    'slot_path': '../data/slot.txt',
    'units': 300,
    'batch_size': 32,
    'max_len': 1000,
    'dropout_rate': 0.2,
    'num_layers': 6,
    'kernel_size': 5,
    'num_patience': 3,
    'lr': 1e-4,
    'max_word_len': 1000,
    'max_char_len': 10,
    'char_embed_size': 100,
    'cnn_filters': 300,
    'cnn_kernel_size': 5,
    'init_lr': 1e-4,
    'max_lr': 8e-4,
}

if __name__ == "__main__":
    word2idx = {}
    with open(params['word_path']) as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            word2idx[line] = i
    params['word2idx'] = word2idx

    intent2idx = {}
    with open(params['intent_path']) as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            intent2idx[line] = i
    params['intent2idx'] = intent2idx

    slot2idx = {}
    with open(params['slot_path']) as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            slot2idx[line] = i
    params['slot2idx'] = slot2idx

    params['word_size'] = len(params['word2idx']) + 1
    params['intent_size'] = len(params['intent2idx']) + 1
    params['slot_size'] = len(params['slot2idx']) + 1

    idx2word = {i: w for w, i in word2idx.items()}
    params['word2idx'] = word2idx
    params['idx2word'] = idx2word
    params['vocab_size'] = len(word2idx) + 1
    model = TransformerModel(params['intent_size'], params['slot_size'], params['lr=1e-4'], params['units'],
                             params['hidden_units'], params['num_heads'], params['multiplier'], params['dropout_rate'],
                             params['num_layers'])
    data = dataset(is_train=1, params=params)

    for words, (intent, slots) in data:
        print("Input shape: {}, {}".format(len(words), len(words[0])))
        y_intent, y_slots = model(words)
        print("Model Output")
        print("Intent shape: {}".format(y_intent.shape))
        print("Slots shape: {}".format(y_slots.shape))
        print('\n')
        break

    print("Fitting model")
    model.fit(data, epochs=2)
    model.save("transformer_model.h5")

    print("Evaluate model")
    model = tf.keras.models.load_model("bidirectional_lstm_model.h5")
    data = dataset(is_train=0, params=params)
    model.evaluate(data)

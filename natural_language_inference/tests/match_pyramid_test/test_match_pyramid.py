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

"""Test case for Text Matching as Image Recognition using Match Pyramid for Natural Language Inference"""

import tensorflow as tf
import os

from Hermes.natural_language_inference.match_pyramid_model.match_pyramid import MatchPyramidModel
from Hermes.natural_language_inference.dataset.loader import dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("TensorFlow Version", tf.__version__)
print('GPU Enabled:', tf.test.is_gpu_available())

params = {
    # 'vocab_path': '../data/word.txt',
    'train_path': '../data/train.txt',
    'test_path': '../data/test.txt',
    'num_samples': 25000,
    'units': 300,
    'num_labels': 3,
    'batch_size': 32,
    'max_len': 1000,
    'dropout_rate': 0.2,
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
    'max_len1': 16,
    'max_len2': 12
}

if __name__ == "__main__":
    _word2idx = tf.keras.datasets.imdb.get_word_index()
    word2idx = {w: i+3 for w, i in _word2idx.items()}
    word2idx['<pad>'] = 0
    word2idx['<start>'] = 1
    word2idx['<unk>'] = 2
    idx2word = {i: w for w, i in word2idx.items()}
    params['word2idx'] = word2idx
    params['idx2word'] = idx2word
    params['vocab_size'] = len(word2idx) + 1
    model = MatchPyramidModel(params['lr'], params['dropout_rate'], params['units'], params['max_len1'],
                              params['max_len2'])
    data = dataset(is_train=1, params=params)

    for x, y in data:
        print("Input shape: {}, {}".format(len(x), len(x[0])))
        print("Target shape: {}".format(len(y)))
        out = model(x)
        print("Output shape: {}".format(out.shape))
        print("Model Output")
        print(out)
        print('\n')
        break

    print("Fitting model")
    model.fit(data, epochs=2)
    model.save("match_pyramid_model.h5")

    print("Evaluate model")
    model = tf.keras.models.load_model("match_pyramid_model.h5")

    data = dataset(is_train=0, params=params)
    model.evaluate(data)

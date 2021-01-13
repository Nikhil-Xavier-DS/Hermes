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

"""Test case for GRU based Sequence to Sequence Model for Semantic Parsing"""

import tensorflow as tf
import os

from Hermes.semantic_parsing.seq2seq_gru_model.seq2seq import Seq2SeqModel
from Hermes.semantic_parsing.dataset.loader import dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("TensorFlow Version", tf.__version__)
print('GPU Enabled:', tf.test.is_gpu_available())

params = {
    'train_path': '../data/train.txt',
    'test_path': '../data/test.txt',
    'target_path': '../data/word.txt',
    'band_width': 12,
    'units': 300,
    'batch_size': 32,
    'max_len': 1000,
    'dropout_rate': 0.2,
    'lr': 1e-4,
    'init_lr': 1e-4,
    'max_lr': 8e-4,
}

if __name__ == "__main__":
    word2idx = {}
    with open(params['target_path']) as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            word2idx[line] = i
    idx2word = {i: w for w, i in word2idx.items()}
    params['word2idx'] = word2idx
    params['idx2word'] = idx2word
    params['vocab_size'] = len(word2idx) + 1
    model = Seq2SeqModel(params['lr'], params['dropout_rate'], params['units'], params['band_width'],
                         params['vocab_size'])
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
    model.save("seq2seq_model.h5")

    print("Evaluate model")
    model = tf.keras.models.load_model("seq2seq_model.h5")

    data = dataset(is_train=0, params=params)
    model.evaluate(data)

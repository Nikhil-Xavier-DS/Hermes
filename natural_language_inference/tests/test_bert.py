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

"""Test case for BERT (Bidirectional Encoder Representations from Transformers) model for Natural Language Inference."""

import tensorflow as tf
import os
from Hermes.natural_language_inference.bert_model.bert import BertInference
from Hermes.natural_language_inference.dataset.loader import bert_dataset
from tensorflow.keras.optimizers import Adam

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("TensorFlow Version", tf.__version__)
print('GPU Enabled:', tf.test.is_gpu_available())

params = {
    'train_path': '../data/train.txt',
    'test_path': '../data/test.txt',
    'num_samples': 25000,
    'num_labels': 2,
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
}

if __name__ == "__main__":
    model = BertInference(params['dropout_rate'], params['units'])
    data = bert_dataset(is_train=1, params=params)

    for x1, x2, y in data:
        print("Input shape: {}, {}".format(len(x), len(x[0])))
        print("Target shape: {}".format(len(y)))
        out = model((x1, x2))
        print("Output shape: {}".format(out.shape))
        print("Model Output")
        print(out)
        print('\n')
        break

    print("Fitting model")
    model.compile(optimizer=Adam(params['lr']), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=.2))
    model.fit(data, epochs=2)
    model.save("bert_model.h5")

    print("Evaluate model")
    model = tf.keras.models.load_model("bert_model.h5")

    data = bert_dataset(is_train=0, params=params)
    model.evaluate(data)

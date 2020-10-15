import tensorflow as tf
import numpy as np

from collections import Counter
from pathlib import Path

params = {
    # 'vocab_path': '../data/word.txt',
    'train_path': '../data/train.txt',
    'test_path': '../data/test.txt',
    'num_samples': 25000,
    'num_labels': 2,
    'batch_size': 32,
    'max_len': 1000,
    'dropout_rate': 0.2,
    'kernel_size': 5,
    'num_patience': 3,
    'lr': 3e-4,
    'max_word_len': 1000,
    'max_char_len': 10,
    'char_embed_size': 100,
    'cnn_filters': 300,
    'cnn_kernel_size': 5,
    'init_lr': 1e-4,
    'max_lr': 8e-4,
}


def write_file(f_path, xs, ys):
    with open(f_path, 'w', encoding="utf-8") as f:
        for x, y in zip(xs, ys):
            f.write(str(y) + '\t' + ' '.join([idx2word[i] for i in x][1:]) + '\n')


def data_generator(f_path, params):
    with open(f_path, encoding="utf8") as f:
        print('Reading', f_path)
        for line in f:
            line = line.rstrip()
            label, text = line.split('\t')
            text = text.split(' ')
            x = [params['word2idx'].get(w, len(params['word2idx'])) for w in text]
            if len(x) > params['max_len']:
                x = x[:params['max_len']]
            y = int(label)
            yield x, y


def char_data_generator(f_path, params):
    with open(f_path, encoding="utf8") as f:
        print('Reading', f_path)
        for line in f:
            line = line.rstrip()
            label, text = line.split('\t')
            text = text.split(' ')
            words = [params['word2idx'].get(w, len(params['word2idx'])) for w in text]
            if len(words) >= params['max_word_len']:
                words = words[:params['max_word_len']]
            chars = []
            for w in text:
                temp = []
                for c in list(w):
                    temp.append(params['char2idx'].get(c, len(params['char2idx'])))
                if len(temp) < params['max_char_len']:
                    temp += [0] * (params['max_char_len'] - len(temp))
                else:
                    temp = temp[:params['max_char_len']]
                chars.append(temp)
            if len(chars) >= params['max_word_len']:
                chars = chars[:params['max_word_len']]
            y = int(label)
            yield words, chars, y


def dataset(is_train, params):
    if is_train:
        data = tf.data.Dataset.from_generator(lambda: data_generator(params['train_path'], params),
                                              output_shapes=([None], ()),
                                              output_types=(tf.int32, tf.int32))
        data = data.shuffle(params['num_samples'])
        data = data.padded_batch(params['batch_size'], ([None], ()), (0, -1))
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        data = tf.data.Dataset.from_generator(lambda: data_generator(params['test_path'], params),
                                              output_shapes=([None], ()),
                                              output_types=(tf.int32, tf.int32))
        data = data.shuffle(params['num_samples'])
        data = data.padded_batch(params['batch_size'], ([None], ()), (0, -1))
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
    return data


def char_dataset(is_train, params):
    if is_train:
        data = tf.data.Dataset.from_generator(lambda: data_generator(params['train_path'], params),
                                              output_shapes=([None], [None, params['max_char_len']], ()),
                                              output_types=(tf.int32, tf.int32, tf.int32))
        data = data.shuffle(params['num_samples'])
        data = data.padded_batch(params['batch_size'], ([None], [None, params['max_char_len']], ()), (0, 0, -1))
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        data = tf.data.Dataset.from_generator(lambda: data_generator(params['test_path'], params),
                                              output_shapes=([None], [None, params['max_char_len']], ()),
                                              output_types=(tf.int32, tf.int32, tf.int32))
        data = data.shuffle(params['num_samples'])
        data = data.padded_batch(params['batch_size'], ([None], [None, params['max_char_len']], ()), (0, 0, -1))
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
    return data


if __name__ == "__main__":
    _word2idx = tf.keras.datasets.imdb.get_word_index()
    word2idx = {w: i + 3 for w, i in _word2idx.items()}
    word2idx['<pad>'] = 0
    word2idx['<start>'] = 1
    word2idx['<unk>'] = 2
    idx2word = {i: w for w, i in word2idx.items()}
    params['word2idx'] = word2idx
    params['idx2word'] = idx2word
    params['vocab_size'] = len(word2idx) + 1

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()

    write_file('../data/train.txt', x_train, y_train)
    write_file('../data/test.txt', x_test, y_test)

    embedding = np.zeros((len(word2idx) + 1, 300))
    with open('../data/glove.840B.300d.txt', encoding="utf-8") as f:
        count = 0
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print('- At line {}'.format(i))
            line = line.rstrip()
            sp = line.split(' ')
            word, vec = sp[0], sp[1:]
            if word in word2idx:
                count += 1
                embedding[word2idx[word]] = np.asarray(vec, dtype='float32')

    print("[%d / %d] words have found pre-trained values" % (count, len(word2idx)))
    np.save('../data/embedding.npy', embedding)
    print('Saved ../data/word.npy')

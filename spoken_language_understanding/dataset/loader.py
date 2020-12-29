import tensorflow as tf
import numpy as np
import re

from collections import Counter
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer, XLNetTokenizer

params = {
    'train_path': '../data/train.txt',
    'test_path': '../data/test.txt',
    'num_samples': 25000,
    'num_labels': 3,
    'batch_size': 32,
    'max_len': 500,
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
    'max_lr': 8e-4
}

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                               lowercase=True,
                                               add_special_tokens=True)

albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2',
                                                   lowercase=True,
                                                   add_special_tokens=True)

roberta_tokenizer = RobertaTokenizer.from_pretrained('robert-base',
                                                     lowercase=True,
                                                     add_special_tokens=True)

xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased',
                                                 lowercase=True,
                                                 add_special_tokens=True)


def data_generator(f_path, params):
    label2idx = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
    with open(f_path) as f:
        print('Reading', f_path)
        for line in f:
            line = line.rstrip()
            label, text1, text2 = line.split('\t')
            if label == '-':
                continue
            text1 = [params['word2idx'].get(w, len(params['word2idx'])) for w in text1]
            text2 = [params['word2idx'].get(w, len(params['word2idx'])) for w in text2]
            yield (text1, text2), label2idx[label]


def bert_data_generator(f_paths, params):
    label2idx = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
    for f_path in f_paths:
        with open(f_path) as f:
            print('Reading', f_path)
            for line in f:
                line = line.rstrip()
                label, text1, text2 = line.split('\t')
                if label == '-':
                    continue
                text1 = bert_tokenizer.tokenize(text1)
                text2 = bert_tokenizer.tokenize(text2)
                if len(text1) + len(text2) + 3 > params['max_len']:
                    _max_len = (params['max_len'] - 3) // 2
                    text1 = text1[:_max_len]
                    text2 = text2[:_max_len]
                text = ['[CLS]'] + text1 + ['[SEP]'] + text2 + ['[SEP]']
                text = bert_tokenizer.convert_tokens_to_ids(text)
                seg = [0] + [0] * len(text1) + [0] + [1] * len(text2) + [1]
                y = label2idx[label]
                yield text, seg, y



def dataset(is_training, params):
    if is_training:
        ds = tf.data.Dataset.from_generator(
            lambda: data_generator(params['train_path'], params['word2idx']),
            output_shapes=(([None], [None]), ()),
            output_types=((tf.int32, tf.int32), tf.int32))
        ds = ds.shuffle(params['buffer_size'])
        ds = ds.padded_batch(params['batch_size'], padded_shapes=(([None], [None]), ()), padding_values=((0, 0), -1))
    else:
        ds = tf.data.Dataset.from_generator(
            lambda: data_generator(params['test_path'], params['word2idx']),
            output_shapes=(([None], [None]), ()),
            output_types=((tf.int32, tf.int32), tf.int32))
        ds = ds.padded_batch(params['batch_size'], padded_shapes=(([None], [None]), ()), padding_values=((0, 0), -1))
    return ds


def albert_dataset(is_train, params):
    if is_train:
        data = tf.data.Dataset.from_generator(lambda: albert_data_generator(params['train_path'], params),
                                              output_shapes=([None], [None], ()),
                                              output_types=(tf.int32, tf.int32, tf.int32))
        data = data.shuffle(params['num_samples'])
        data = data.padded_batch(params['batch_size'], ([None], [None], ()), (0, 0, -1))
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        data = tf.data.Dataset.from_generator(lambda: albert_data_generator(params['test_path'], params),
                                              output_shapes=([None], [None], ()),
                                              output_types=(tf.int32, tf.int32, tf.int32))
        data = data.shuffle(params['num_samples'])
        data = data.padded_batch(params['batch_size'], ([None], [None], ()), (0, 0, -1))
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
    return data


def bert_dataset(is_train, params):
    if is_train:
        data = tf.data.Dataset.from_generator(lambda: bert_data_generator(params['train_path'], params),
                                              output_shapes=([None], [None], ()),
                                              output_types=(tf.int32, tf.int32, tf.int32))
        data = data.shuffle(params['num_samples'])
        data = data.padded_batch(params['batch_size'], ([None], [None], ()), (0, 0, -1))
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        data = tf.data.Dataset.from_generator(lambda: bert_data_generator(params['test_path'], params),
                                              output_shapes=([None], [None], ()),
                                              output_types=(tf.int32, tf.int32, tf.int32))
        data = data.shuffle(params['num_samples'])
        data = data.padded_batch(params['batch_size'], ([None], [None], ()), (0, 0, -1))
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
    return data


def roberta_dataset(is_train, params):
    if is_train:
        data = tf.data.Dataset.from_generator(lambda: roberta_data_generator(params['train_path'], params),
                                              output_shapes=([None], [None], ()),
                                              output_types=(tf.int32, tf.int32, tf.int32))
        data = data.shuffle(params['num_samples'])
        data = data.padded_batch(params['batch_size'], ([None], [None], ()), (0, 0, -1))
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        data = tf.data.Dataset.from_generator(lambda: roberta_data_generator(params['test_path'], params),
                                              output_shapes=([None], [None], ()),
                                              output_types=(tf.int32, tf.int32, tf.int32))
        data = data.shuffle(params['num_samples'])
        data = data.padded_batch(params['batch_size'], ([None], [None], ()), (0, 0, -1))
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
    return data


def xlnet_dataset(is_train, params):
    if is_train:
        data = tf.data.Dataset.from_generator(lambda: xlnet_data_generator(params['train_path'], params),
                                              output_shapes=([None], [None], ()),
                                              output_types=(tf.int32, tf.int32, tf.int32))
        data = data.shuffle(params['num_samples'])
        data = data.padded_batch(params['batch_size'], ([None], [None], ()), (0, 0, -1))
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        data = tf.data.Dataset.from_generator(lambda: xlnet_data_generator(params['test_path'], params),
                                              output_shapes=([None], [None], ()),
                                              output_types=(tf.int32, tf.int32, tf.int32))
        data = data.shuffle(params['num_samples'])
        data = data.padded_batch(params['batch_size'], ([None], [None], ()), (0, 0, -1))
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
    return data


def normalize(x):
    x = x.lower()
    x = x.replace('.', '')
    x = x.replace(',', '')
    x = x.replace(';', '')
    x = x.replace('!', '')
    x = x.replace('#', '')
    x = x.replace('(', '')
    x = x.replace(')', '')
    x = x.replace(':', '')
    x = x.replace('%', '')
    x = x.replace('&', '')
    x = x.replace('$', '')
    x = x.replace('?', '')
    x = x.replace('"', '')
    x = x.replace('/', ' ')
    x = x.replace('-', ' ')
    x = x.replace("n't", " n't ")
    x = x.replace("'", " ' ")
    x = re.sub(r'\d+', ' <num> ', x)
    x = re.sub(r'\s+', ' ', x)
    return x


def write_text(in_path, out_path):
    with open(in_path) as f_in, open(out_path, 'w') as f_out:
        f_in.readline()
        for line in f_in:
            line = line.rstrip()
            sp = line.split('\t')
            label, sent1, sent2 = sp[0], sp[5], sp[6]
            sent1 = normalize(sent1)
            sent2 = normalize(sent2)
            f_out.write(label + '\t' + sent1 + '\t' + sent2 + '\n')


def norm_weight(inp, out, scale=0.01):
    W = scale * np.random.randn(inp, out)
    return W.astype(np.float32)


if __name__ == "__main__":
    write_text('../data/snli_data/snli_1.0_train.txt', '../data/train.txt')
    write_text('../data/snli_data/snli_1.0_test.txt', '../data/test.txt')

    counter = Counter()
    with open('../data/train.txt') as f:
        for line in f:
            line = line.rstrip()
            label, sent1, sent2 = line.split('\t')
            counter.update(sent1.split())
            counter.update(sent2.split())
    words = [w for w, freq in counter.most_common() if freq >= 3]

    word2idx = dict()
    word2idx['<pad>'] = 0
    for i, word in enumerate(words):
        word = word.rstrip()
        word2idx[word] = i

    embedding = np.zeros((len(word2idx) + 1, 300))
    with open('../data/glove.840B.300d.txt', encoding="utf-8") as f:
        count = 0
        for i, line in enumerate(f):
            line = line.rstrip()
            sp = line.split(' ')
            word, vec = sp[0], sp[1:]
            if word in word2idx:
                count += 1
                embedding[word2idx[word]] = np.asarray(vec, dtype='float32')
    np.save('../data/embedding.npy', embedding)

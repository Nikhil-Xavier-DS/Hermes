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

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base',
                                                     lowercase=True,
                                                     add_special_tokens=True)

xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased',
                                                 lowercase=True,
                                                 add_special_tokens=True)


def data_generator(f_path, params):
    with open(f_path) as f:
        for line in f:
            line = line.rstrip()
            text, slot_intent = line.split('\t')
            words = text.split()[1:-1]
            slot_intent = slot_intent.split()
            slots, intent = slot_intent[1:-1], slot_intent[-1]
            words = [params['word2idx'].get(w, len(params['word2idx'])) for w in words]
            intent = params['intent2idx'].get(intent, len(params['intent2idx']))
            slots = [params['slot2idx'].get(s, len(params['slot2idx'])) for s in slots]
            yield words, (intent, slots)


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
    _shapes = ([None], ((), [None]))
    _types = (tf.int32, (tf.int32, tf.int32))
    _pads = (0, (-1, 0))

    if is_training:
        ds = tf.data.Dataset.from_generator(
            lambda: data_generator(params['train_path'], params),
            output_shapes=([None], ((), [None])),
            output_types=(tf.int32, (tf.int32, tf.int32)))
        ds = ds.shuffle(params['num_samples'])
        ds = ds.padded_batch(params['batch_size'], ([None], ((), [None])), (0, (-1, 0)))
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_generator(
            lambda: data_generator(params['test_path'], params),
            output_shapes=([None], ((), [None])),
            output_types=(tf.int32, (tf.int32, tf.int32)))
        ds = ds.padded_batch(params['batch_size'], ([None], ((), [None])), (0, (-1, 0)))
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


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


if __name__ == "__main__":
    counter_word = Counter()
    counter_intent = Counter()
    counter_slot = Counter()
    with open('../data/atis_data/atis.train.w-intent.iob') as f:
        for line in f:
            line = line.rstrip()
            text, slot_intent = line.split('\t')
            print(f"text: {text}")
            print(f"slot_intent: {slot_intent}")
            words = text.split()[1:-1]
            words = ['<digit>' if str.isdigit(w) else w for w in words]
            print(f"words: {words}")
            slot_intent = slot_intent.split()
            slots, intent = slot_intent[1:-1], slot_intent[-1]
            print(f"slots: {slots}")
            print(f"intent: {intent}")
            counter_word.update(words)
            counter_intent.update([intent])
            counter_slot.update(slots)


    def freq_func(x):
        return [w for w, freq in x.most_common()]


    words = ['<pad>'] + freq_func(counter_word)
    intents = freq_func(counter_intent)
    slots = freq_func(counter_slot)

    for vocab_li, path in zip([words, intents, slots], ['../data/word.txt', '../data/intent.txt', '../data/slot.txt']):
        with open(path, 'w') as f:
            for w in vocab_li:
                f.write(w + '\n')

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

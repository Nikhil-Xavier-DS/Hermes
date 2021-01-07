import tensorflow as tf
import numpy as np

from collections import Counter

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


def data_generator(path, params):
    with open(path) as f:
        for line in f:
            text_raw, text_tokenized, label = line.split('\t')
            text_tokenized = text_tokenized.lower().split()
            label = label.replace('[', '[ ').lower().split()
            source = [params['tgt2idx'].get(w, len(params['tgt2idx'])) for w in text_tokenized]
            target = [params['tgt2idx'].get(w, len(params['tgt2idx'])) for w in label]
            target_in = [1] + target
            target_out = target + [2]
            yield source, target_in, target_out


def dataset(is_train, params):
    if is_train:
        ds = tf.data.Dataset.from_generator(lambda: data_generator(params['train_path'], params),
                                            output_shapes=([None], [None], [None]),
                                            output_types=(tf.int32, tf.int32, tf.int32))
        ds = ds.shuffle(params['buffer_size'])
        ds = ds.padded_batch(params['batch_size'],
                             ([None], [None], [None]),
                             (0, 0, 0))
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_generator(lambda: data_generator(params['test_path'], params),
                                            output_shapes=([None], [None], [None]),
                                            output_types=(tf.int32, tf.int32, tf.int32))
        ds = ds.padded_batch(params['batch_size'],
                             ([None], [None], [None]),
                             (0, 0, 0))
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


if __name__ == "__main__":
    encoder_counter = Counter()
    decoder_counter = Counter()

    with open('../data/train.tsv') as f:
        for line in f:
            line = line.rstrip()
            text_raw, text_tokenized, label = line.split('\t')
            encoder_counter.update(text_tokenized.lower().split())
            decoder_counter.update(label.replace('[', '[ ').lower().split())

    with open('../data/source.txt', 'w') as f:
        f.write('<pad>\n')
        for (w, freq) in encoder_counter.most_common():
            f.write(w + '\n')

    with open('../data/target.txt', 'w') as f:
        f.write('<pad>\n')
        f.write('<start>\n')
        f.write('<end>\n')
        for (w, freq) in decoder_counter.most_common():
            f.write(w + '\n')

    words = [w for w, freq in decoder_counter.most_common() if freq >= 3]
    word2idx = dict()
    word2idx['<pad>'] = 0
    word2idx['<start>'] = 1
    word2idx['<end>'] = 2
    for i, word in enumerate(words):
        word = word.rstrip()
        word2idx[word] = i + 3

    embedding = np.zeros((len(word2idx) + 1, 300))
    with open('../data/glove.840B.300d.txt', encoding="utf-8") as f:
        count = 0
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print('At line {}'.format(i))
            line = line.rstrip()
            sp = line.split(' ')
            word, vec = sp[0], sp[1:]
            if word in word2idx:
                count += 1
                embedding[word2idx[word]] = np.asarray(vec, dtype='float32')

    print("[%d / %d] words have found pre-trained values" % (count, len(word2idx)))
    np.save('../data/embedding.npy', embedding)

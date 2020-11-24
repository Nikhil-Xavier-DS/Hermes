import numpy as np
import re

from collections import Counter


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
    # word2idx = {w: i + 3 for w, i in _word2idx.items()}

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

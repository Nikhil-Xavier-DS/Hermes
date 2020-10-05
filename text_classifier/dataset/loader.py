params = {
    'vocab_path': '../vocab/word.txt',
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
}

word2idx = {}
with open(params['vocab_path']) as f:
    for i, line in enumerate(f):
        line = line.rstrip()
        word2idx[line] = i
params['word2idx'] = word2idx
params['vocab_size'] = len(word2idx) + 1


def data_generator(f_path, params):
    with open(f_path) as f:
        print('Reading', f_path)
        for line in f:
            line = line.rstrip()
            label, text = line.split('\t')
            text = text.split(' ')
            x = [params['word2idx'].get(w, len(word2idx)) for w in text]
            if len(x) > params['max_len']:
                x = x[:params['max_len']]
            y = int(label)
            yield x, y

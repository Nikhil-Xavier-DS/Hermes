import tensorflow as tf
import os

# from Hermes.text_classifier.feedforward_attention_model.basic_attention_model import FeedForwardAttention
from Hermes.text_classifier.feedforward_attention_model.cnn_attention_model import FeedForwardAttention
from Hermes.text_classifier.dataset.loader import dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("TensorFlow Version", tf.__version__)
print('GPU Enabled:', tf.test.is_gpu_available())

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
    model = FeedForwardAttention(params)
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
    model.save("feed_forward_model.h5")

    print("Evaluate model")
    model = tf.keras.models.load_model("feed_forward_model.h5")

    data = dataset(is_train=0, params=params)
    model.evaluate(data)

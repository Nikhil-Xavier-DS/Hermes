import tensorflow as tf
import time
import logging
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D
from tensorflow.keras.optimizers import Adam


class KernelAttentivePooling(Model):
    def __init__(self, params):
        super().__init__()
        self.dropout = Dropout(params['dropout_rate'])
        self.kernel = Conv1D(filters=1,
                             kernel_size=params['kernel_size'],
                             padding='same',
                             activation=tf.tanh,
                             use_bias=True)

    def call(self, inputs, training=False):
        x, masks = inputs
        x = self.dropout(x, training=training)
        x = self.kernel(x)
        align = tf.squeeze(x, -1)
        padding = tf.fill(tf.shape(align), float('-inf'))
        align = tf.where(tf.equal(masks, 0), padding, align)
        align = tf.nn.softmax(align)
        align = tf.expand_dims(align, -1)
        return tf.squeeze(tf.matmul(x, align, transpose_a=True), axis=-1)


class FeedForwardAttention(Model):
    EPOCHS = 1
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.INFO)

    def __init__(self, params):
        super().__init__()
        self.embedding = tf.Variable(np.load('../data/embedding.npy'),
                                     dtype=tf.float32,
                                     name='pretrained_embedding')
        self.attentive_pooling = KernelAttentivePooling(params)
        self.out_linear = Dense(2)
        self.optimizer = Adam(params['lr'])
        self.accuracy = tf.keras.metrics.Accuracy()
        self.decay_lr = tf.optimizers.schedules.ExponentialDecay(params['lr'], 1000, 0.95)
        self.params = params
        self.logger = logging.getLogger('tensorflow')
        self.logger.setLevel(logging.INFO)

    def call(self, inputs, training=False):
        if inputs.dtype != tf.int32:
            inputs = tf.cast(inputs, tf.int32)
        masks = tf.sign(inputs)
        x = tf.nn.embedding_lookup(self.embedding, inputs)
        x = self.attentive_pooling((x, masks), training=training)
        x = self.out_linear(x)
        return x

    def fit(self, data, epochs=EPOCHS):
        t0 = time.time()
        step = 0
        epoch = 1
        while epoch <= epochs:
            for texts, labels in data:
                with tf.GradientTape() as tape:
                    logits = self.call(texts, training=True)
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
                self.optimizer.lr.assign(self.decay_lr(step))
                grads = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                if step % 100 == 0:
                    self.logger.info("Step {} | Loss: {:.4f} | Spent: {:.1f} secs | LR: {:.6f}".format(
                        step, loss.numpy().item(), time.time() - t0, self.optimizer.lr.numpy().item()))
                    t0 = time.time()
                step += 1
            epoch += 1
        return True

    def evaluate(self, data):
        self.accuracy.reset_states()
        for texts, labels in data:
            logits = self.call(texts, training=False)
            y_pred = tf.argmax(logits, axis=-1)
            self.accuracy.update_state(y_true=labels, y_pred=y_pred)

        accuracy = self.accuracy.result().numpy()
        self.logger.info("Evaluation Accuracy: {:.3f}".format(accuracy))
        self.logger.info("Accuracy: {:.3f}".format(accuracy))

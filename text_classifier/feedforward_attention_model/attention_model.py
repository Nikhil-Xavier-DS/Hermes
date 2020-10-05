import tensorflow as tf
from tensorflow.keras import Model
from tensorflow import optimizers
import time
from Hermes.text_classifier.neural_translation_attention_model.attention_model.attention import \
    BahdanauAttention
from Hermes.neural_machine_translation.neural_translation_attention_model.encoder_model.encoder import \
    HybridBLSTMEncoder
from Hermes.neural_machine_translation.neural_translation_attention_model.decoder_model.decoder import LSTMDecoder


class KernelAttentivePooling(Model):
    def __init__(self, params):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(params['dropout_rate'])
        self.kernel = tf.keras.layers.Dense(units=1,
                                            activation=tf.tanh,
                                            use_bias=False)

    def call(self, inputs, training=False):
        x, masks = inputs
        align = tf.squeeze(self.kernel(self.dropout(x, training=training)), -1)
        paddings = tf.fill(tf.shape(align), float('-inf'))
        align = tf.where(tf.equal(masks, 0), paddings, align)
        align = tf.nn.softmax(align)
        align = tf.expand_dims(align, -1)
        return tf.squeeze(tf.matmul(x, align, transpose_a=True), -1)


class FeedForwardAttention(tf.keras.Model):
    def __init__(self, params):
        super().__init__()
        self.embedding = tf.Variable(np.load('../vocab/word.npy'),
                                     dtype=tf.float32,
                                     name='pretrained_embedding')
        self.attentive_pooling = KernelAttentivePooling(params)
        self.out_linear = tf.keras.layers.Dense(2)
        self.optimizer = Adam(params['lr'])
        self.accuracy = tf.keras.metrics.Accuracy()
        self.decay_lr = tf.optimizers.schedules.ExponentialDecay(params['lr'], 1000, 0.95)
        self.params = params
        logger = logging.getLogger('tensorflow')
        logger.setLevel(logging.INFO)

    def call(self, inputs, training=False):
        if inputs.dtype != tf.int32:
            inputs = tf.cast(inputs, tf.int32)
        masks = tf.sign(inputs)
        x = tf.nn.embedding_lookup(self.embedding, inputs)
        x = self.attentive_pooling((x, masks), training=training)
        x = self.out_linear(x)
        return x

    def fit(self, data_generator, epochs=EPOCHS):
        t0 = time.time()
        step = 0
        epoch = 1
        while epoch >= epochs:
            ds = tf.data.Dataset.from_generator(lambda: data_generator(self.params['train_path'], params),
                                                output_shapes=([None], ()),
                                                output_types=(tf.int32, tf.int32))
            ds = ds.shuffle(self.params['num_samples'])
            ds = ds.padded_batch(self.params['batch_size'], ([None], ()), (0, -1))
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

            for texts, labels in ds:
                with tf.GradientTape() as tape:
                    logits = self.call(texts, training=True)
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
                self.optimizer.lr.assign(self.decay_lr(step))
                grads = tape.gradient(loss, model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                if step % 50 == 0:
                    logger.info("Step {} | Loss: {:.4f} | Spent: {:.1f} secs | LR: {:.6f}".format(
                        step, loss.numpy().item(), time.time() - t0, self.optimizer.lr.numpy().item()))
                    t0 = time.time()
                step += 1

            epoch += 1
        return self.history

    def evaluate(self, data_generator):
            self.accuracy.reset_states()
            ds = tf.data.Dataset.from_generator(lambda: data_generator(self.params['test_path'], params),
                                                output_shapes=([None], ()),
                                                output_types=(tf.int32, tf.int32))
            ds = ds.padded_batch(self.params['batch_size'], ([None], ()), (0, -1))
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

            for texts, labels in ds:
                logits = self.call(texts, training=False)
                y_pred = tf.argmax(logits, axis=-1)
                self.accuracy.update_state(y_true=labels, y_pred=y_pred)

            accuracy = self.accuracy.result().numpy()
            logger.info("Evaluation Accuracy: {:.3f}".format(accuracy))
            history_acc.append(accuracy)
            logger.info("Accuracy: {:.3f}".format(accuracy))

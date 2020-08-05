import tensorflow as tf
from tensorflow.keras import Model
from Hermes.neural_machine_translation.neural_translation_attention_model.attention_model.attention import BahdanauAttention
from Hermes.neural_machine_translation.neural_translation_attention_model.encoder_model.encoder import HybridBLSTMEncoder
from Hermes.neural_machine_translation.neural_translation_attention_model.decoder_model.decoder import LSTMDecoder

MAX_LENGTH_TARGET = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
EPOCHS = 1


class SeqToSeqHybridBLSTM(Model):
    """
    Create Hybrid BLSTM LSTM based Encoder Decoder Sequence to Sequence model.
    --------------------------------------------------------------------------
    Inputs:

    """

    def __init__(self, batch_size=BATCH_SIZE, **kwargs):
        super(SeqToSeqHybridBLSTM, self).__init__(name="SeqToSeq Hybrid BLSTM LSTM", **kwargs)
        required_fields = ['vocab_size', 'vocab_size_out', 'embedding_dim', 'enc_units', 'dec_units', 'word_id',
                           'word_id_out']
        self.batch_size = batch_size
        self.enc_units = kwargs['enc_units']
        self.dec_units = kwargs['dec_units']
        if not all(name in kwargs for name in required_fields):
            raise Exception("Requires ", ",".join(required_fields))
        else:
            self.encoder = HybridBLSTMEncoder(kwargs['vocab_size'], kwargs['embedding_dim'], kwargs['enc_units'], batch_size)
            if 'att_units' in kwargs:
                self.att_units = kwargs['att_units']
            else:
                self.att_units = self.dec_units
            self.attention = BahdanauAttention(self.att_units)
            self.decoder = LSTMDecoder(kwargs['vocab_size'], kwargs['embedding_dim'], kwargs['dec_units'],
                                       self.att_units, batch_size)
            self.word_id = kwargs["word_id"]
            self.word_id_out = kwargs["word_id_out"]
            if "id_word" in kwargs:
                self.id_word = kwargs["id_word"]
            else:
                self.id_word = {symbol: i for i, symbol in self.word_id.items()}
            if "id_word_out" in kwargs:
                self.id_word_out = kwargs["id_word_out"]
            else:
                self.id_word_out = {symbol: i for i, symbol in self.word_id_out.items()}
            if "max_len_targ" in kwargs:
                self.max_len_tar = kwargs["max_len_targ"]
            else:
                self.max_len_tar = MAX_LENGTH_TARGET
            if "learning_rate" in kwargs:
                self.max_len_tar = kwargs["learning_rate"]
            else:
                self.max_len_tar = LEARNING_RATE
        self.optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        self.loss_object = MaskedLoss()

    def call(self, inp, **kwargs):
        enc_hidden_h, enc_hidden_c = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden_h, enc_hidden_c = self.encoder(inp, enc_hidden_h, enc_hidden_c)
        dec_hidden_h = enc_hidden_h
        dec_hidden_c = enc_hidden_c
        predictions = []
        for i in range(self.batch_size):
            predictions.append([])
        dec_input = tf.expand_dims([self.word_id_out['start']] * self.batch_size, 1)
        dec_input = tf.reshape(dec_input, (self.batch_size, -1))
        prediction = dec_input
        for target in range(self.max_len_tar):
            prediction, dec_hidden_h, dec_hidden_c, attention_weights = self.decoder(dec_input, dec_hidden_h,
                                                                                     dec_hidden_c, enc_output)
            predicted_id = tf.argmax(prediction, 1).numpy()
            for i in range(self.batch_size):
                try:
                    if predictions[i][-1] == self.word_id_out['<end>']:
                        pass
                    elif predicted_id[i] == self.word_id_out['<end>']:
                        predictions[i].append(self.word_id_out['<end>'])
                    else:
                        predictions[i].append(predicted_id[i])
                except:
                    if predicted_id[i] == self.word_id_out['<end>']:
                        predictions[i].append(self.word_id_out['<end>'])
                    else:
                        predictions[i].append(predicted_id[i])
            dec_input = tf.convert_to_tensor(predicted_id)
            dec_input = tf.reshape(dec_input, self.batch_size, -1)
        return predictions

    def predict(self, inp, **kwargs):
        return self.call(inp, **kwargs)

    def fit(self, inp, target, epochs=EPOCHS, batch_size=BATCH_SIZE, **kwargs):
        self.compile(optimizer=self.optimizer, loss=self.loss_object)
        return super(Model).fit(inp, target, epochs=epochs, batch_size=batch_size)
        return self.history


class MaskedLoss:
    def __init__(self):
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def call(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

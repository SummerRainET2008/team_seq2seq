import sys
sys.path.append("E:/Paii")
from nmt_with_attention_tf2 import *

import tensorflow as tf
import io
from model import Encoder, Decoder
from utils import convert_data, max_length
from trainer import Trainer


class Predictor(object):
    def __init__(self, param):
        self.param = param
        lines = io.open(param.test, encoding='UTF-8').read().strip().split('\n')
        sentences = [w.strip() for l in lines for w in l.split('\t')]
        test_tensor = convert_data(param.encoder_src, sentences)
        self.max_length = max_length(test_tensor)

    def load_model(self, param):
        self.encoder = Encoder(param.vocab_src_size, param.embedding_dim,
                               param.hidden_units, param.batch_size, param.enc_layers, param.rnn_type)
        self.decoder = Decoder(param.vocab_src_size, param.embedding_dim,
                               param.hidden_units, param.batch_size, param.rnn_type)
        optimizer = tf.keras.optimizers.Adam(lr=0.01)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=self.encoder, decoder=self.decoder)
        checkpoint.restore(tf.train.latest_checkpoint(param.checkpoint_dir))
        print('Restored checkpoint from {}'.format(tf.train.latest_checkpoint(param.checkpoint_dir)))

    # def translate_one(self, sentence):
    #     return Trainer.translate_one(sentence, self.encoder, self.decoder, self.param, self.max_length, self.max_length)

    def evaluate(self):
        return Trainer.evaluate(self.encoder, self.decoder, self.param, self.max_length, self.max_length, valid=False)

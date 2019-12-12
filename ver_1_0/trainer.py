from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append("E:/Paii")

import tensorflow as tf
import numpy as np
import os
import time

from model import Encoder, Decoder
from dataset import to_tf_dataset
from utils import convert_data, generate_src_ref_list

from nmt_with_attention_tf2 import *
from ver_1_2_3.common.bleu.compute_bleu_by_nist_format import compute_bleu_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


class Trainer(object):
    def __init__(self, param):
        self.param = param
        # define optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=0.01)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def train(self, param):
        # get training dataset
        if not tf.test.is_gpu_available(): print('using CPU')
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = param.GPU
        dataset, BUFFER_SIZE, self.max_length_src, self.max_length_trg = \
            to_tf_dataset(param.train, param.batch_size, param.encoder_src, param.encoder_trg)
        # <BatchDataset shapes: ((64, 101), (64, 168)), types: (tf.int32, tf.int32)>
        steps_per_epoch = BUFFER_SIZE // param.batch_size

        # initialize model
        self.encoder = Encoder(param.vocab_src_size, param.embedding_dim,
                               param.hidden_units, param.batch_size,
                               param.enc_layers, param.rnn_type)
        self.decoder = Decoder(param.vocab_trg_size, param.embedding_dim,
                               param.hidden_units, param.batch_size, param.rnn_type)

        # checkpoint
        os.system('rm {}/*'.format(param.checkpoint_dir))
        checkpoint_prefix = os.path.join(param.checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)

        best_bleu = 0
        for epoch in range(param.epochs):
            start = time.time()
            enc_hidden_init = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch_idx, (src, trg)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(src, trg, enc_hidden_init)
                total_loss += batch_loss

                # doing evaluation every evaluate_frequency
                if (batch_idx + 1) % param.evaluate_frequency == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch_idx, batch_loss.numpy()))
                    bleu = Trainer.evaluate(self.encoder, self.decoder, param,
                                            self.max_length_src, self.max_length_trg,  epoch, batch_idx)
                    if best_bleu < bleu:
                        best_bleu = bleu
                        checkpoint.save(file_prefix=checkpoint_prefix)
                        print('Epoch_%s batch_%s, saving checkpoint to '
                              'training_checkpoints dir.' % (epoch, batch_idx))


            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    def train_step(self, src, trg, enc_hidden):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(src, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([SOS] * self.param.batch_size, 1)  # encoder_trg.encode('<s>')

            # Teacher forcing - feeding the target as the next input
            for t in range(1, trg.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                loss += self.loss_function(trg[:, t], predictions)  # difference between trg and pre at t-th time step

                # using teacher forcing
                dec_input = tf.expand_dims(trg[:, t], 1)

        batch_loss = (loss / int(trg.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    @staticmethod
    def translate_one(sentence, encoder_model, decoder_model, param, max_length_src, max_length_trg):
        # sentence: str with blanks eg:'我 不 知道'
        sentence = sentence.strip()
        inputs = convert_data(param.encoder_src, [sentence])
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_length_src, padding='post')
        inputs = tf.convert_to_tensor(inputs)

        hidden = [tf.zeros((1, param.hidden_units))]
        enc_out, enc_hidden = encoder_model(inputs, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([SOS], 0)

        ids = []
        for t in range(max_length_trg):
            predictions, dec_hidden, attention_weights = decoder_model(dec_input, dec_hidden, enc_out)
            predicted_id = int(tf.argmax(predictions[0]).numpy())
            if predicted_id == EOS: break
            ids.append(predicted_id)
            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        result = param.encoder_trg.decode(ids)  # result += encoder_trg.decode([predicted_id]) + ' '
        return result.rstrip()  # string

    @staticmethod
    def evaluate(encoder_model, decoder_model, param, max_length_src,
                 max_length_trg, epoch=None, batch_idx=None, valid=True):
        start_time = time.time()

        if valid:
            file_to_trans = param.valid
            trg = param.valid_trg
        else:
            file_to_trans = param.test
            trg = param.test_trg

        # move outside than it is not needed to calculate every evaluation
        src_list, ref_list = generate_src_ref_list(file_to_trans, trg)
        hyp_list = []
        for src in src_list:
            hyp = Trainer.translate_one(src, encoder_model, decoder_model, param, max_length_src, max_length_trg)
            hyp_list.append(hyp.split())

        elapsed = time.time() - start_time

        # sampling
        ix = np.random.randint(0, len(src_list))
        print('translation sample:')
        print('source sentence is {}'.format(src_list[ix]))
        print('reference sentences are {}'.format(ref_list[ix]))
        print('machine translation is {}'.format(hyp_list[ix]))

        dirname = file_to_trans.split('/')[-1].split('.')[0]
        bleu = corpus_bleu(ref_list, hyp_list, smoothing_function=SmoothingFunction().method1)
        if epoch:
            print('BLEU score for epoch_{}, batch_{} is {}, elapsed time for evaluation '
                  'is {} seconds'.format(epoch, batch_idx, bleu, elapsed))
        else:
            print('BLEU score for test set {} is {}'.format(dirname, bleu))
        return bleu
        # score_list.append((bleu, batch_idx, epoch))

    @staticmethod
    def evaluate_0(encoder_model, decoder_model, param, max_length_src,
                   max_length_trg, epoch=None, batch_idx=None, valid=True):
        start_time = time.time()
        all_to_export = []
        if valid:
            file_to_trans = param.valid
            src = param.valid_src
            ref = param.valid_ref
        else:
            file_to_trans = param.test
            src = param.test_src
            ref = param.test_ref

        with open(file_to_trans, 'r', encoding='utf8') as f:
            for line in f:
                to_export = dict()
                src = line.strip('\n')
                to_export['ch'] = src
                to_export['nbest'] = []
                to_export['nbest'].append((None,
                                           Trainer.translate_one(src, encoder_model, decoder_model,
                                                                 param, max_length_src, max_length_trg)))
                all_to_export.append(to_export)
        elapsed = time.time() - start_time
        dirname = file_to_trans.split('/')[-1].split('.')[0]
        if not os.path.exists('generation/%s' % dirname):
            os.mkdir('generation/%s' % dirname)
        path_to_pydict = 'generation/%s/%s-%s' % (dirname, epoch, batch_idx) + '.nbest.pydict'
        with open(path_to_pydict, 'w', encoding='utf8') as f:
            f.writelines("%s\n" % item for item in all_to_export)
        print('{} saved!'.format(path_to_pydict))

        bleu, _ = compute_bleu_score(src, ref, path_to_pydict)
        if epoch:
            print('BLEU score for epoch_{}, batch_{} is {}, '
                  'elapsed time for evaluation is {} seconds'.format(epoch, batch_idx, bleu, elapsed))
        else:
            print('BLEU score for test set {} is {}'.format(dirname, bleu))
        return bleu
        # score_list.append((bleu, batch_idx, epoch))

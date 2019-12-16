import os
from common.subword.sentence_encoder import SentenceEncoder
from pa_nlp.tf_2x.estimator.param import ParamBase
from pa_nlp.tf_2x import *

class Param(ParamBase):
  def __init__(self):
    super(Param, self).__init__("ver_2_1")

    self.path_bleu = f"{self.path_work}/bleu"
    nlp.ensure_folder_exists(self.path_bleu)

    # data
    self.train: str = f"{self.path_feat}/train.tfrecord"
    self.vali_file: str = f"{self.path_feat}/vali.tfrecord"
    self.test_files = [] #[f"{self.path_feat}/test1.tfrecord"]

    # vocabulary
    # src here refers to Chinese vocab, different from that in train and evaluate
    self.path_to_vob_src = 'corpus/vob.src'
    self.path_to_vob_trg = 'corpus/vob.tgt'
    self.encoder_cn = SentenceEncoder(self.path_to_vob_src)
    self.encoder_en = SentenceEncoder(self.path_to_vob_trg)
    self.vocab_cn_size = 30000
    self.vocab_en_size = 30000

    self.use_polynormial_decay = False
    self.use_warmup = False

    # model
    self.rnn_type = 'LSTM'
    self.embedding_dim = 256 # 256
    self.enc_units = 256 # 1024
    self.dec_units = 256 # 1024
    self.max_length_src = 50
    self.max_length_trg = 50
    self.enc_layers = 4
    self.lr = 5e-4
    self.epoch_num   = 100
    self.batch_size  = 2
    self.evaluate_freq = 10

    '''
    self.train_files = [f"{self.path_feat}/train.tfrecord"]
    self.pretrained_model = os.path.expanduser(
      "~/pretrained_models/bert_data/uncased_L-12_H-768_A-12"
    )
    '''
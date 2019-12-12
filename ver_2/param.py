import os
from common.subword.sentence_encoder import SentenceEncoder
from pa_nlp.tf_2x.estimator.param import ParamBase
from pa_nlp.tf_2x import *

# parameters
# which optimizer
# "decay_learning_rate"
# "dropout_keep_prob": dropout_keep_prob,
# "l2_reg_lambda": l2_reg_lambda,

class Param(ParamBase):
  def __init__(self):
    super(Param, self).__init__("ver_2")

    self.path_bleu = f"{self.path_work}/bleu"
    nlp.ensure_folder_exists(self.path_bleu)

    # data
    self.train: str = f"{self.path_feat}/train.tfrecord"
    self.vali_file: str = f"{self.path_feat}/vali.tfrecord"

    # self.test_file: str = 'corpus/nist05_test.cn'
    self.test_files = [f"{self.path_feat}/test1.tfrecord"]
    self.test_trg = ['corpus/nist05_test.en0', 'corpus/nist05_test.en1',
                     'corpus/nist05_test.en2', 'corpus/nist05_test.en3']
    self.test_src = 'corpus/nist02_test.src'
    self.test_ref = 'corpus/nist02_test.src'

    # vocabulary
    # src here refers to Chinese vocab, different from that in train and evaluate
    self.path_to_vob_src = 'corpus/vob.src'
    self.path_to_vob_trg = 'corpus/vob.tgt'
    self.encoder_cn = SentenceEncoder(self.path_to_vob_src)
    self.encoder_en = SentenceEncoder(self.path_to_vob_trg)
    self.vocab_cn_size = 30000
    self.vocab_en_size = 30000

    # model
    self.rnn_type = 'LSTM'
    self.embedding_dim = 256 # 256
    self.enc_units = 256 # 1024
    self.dec_units = 256 # 1024
    self.max_length_src = 100
    self.max_length_trg = 50
    self.enc_layers = 4
    self.lr = 5e-4
    # self.lr_decay
    self.epoch_num   = 50
    self.batch_size  = 2
    self.evaluate_freq = 10
    # self.atten_dropout = 0.05
    # self.relu_dropout = 0.05
    # self.layer_postprocess_dropout = 0.05
    self.GPU: int = '1'  # which_GPU_to_run: [0, 4), and -1 denote CPU.


    '''
    self.train_files = [f"{self.path_feat}/train.tfrecord"]
    self.vali_file = f"{self.path_feat}/test.0.tfrecord"
    self.test_files = [f"{self.path_feat}/test.1.tfrecord",]
    self.pretrained_model = os.path.expanduser(
      "~/pretrained_models/bert_data/uncased_L-12_H-768_A-12"
    )
        self.train_files = [f"{self.path_feat}/train.tfrecord"]
    self.vali_file = f"{self.path_feat}/test.0.tfrecord"
    self.test_files = [f"{self.path_feat}/test.1.tfrecord",]
    self.pretrained_model = os.path.expanduser(
      "~/pretrained_models/bert_data/uncased_L-12_H-768_A-12"
    )
    '''
import os
from ver_1_2_3.common.subword.sentence_encoder import SentenceEncoder

# parameters
# which optimizer
# "learning_rate": learning_rate,
# "dropout_keep_prob": dropout_keep_prob,
# "l2_reg_lambda": l2_reg_lambda,

class Param(object):
    def __init__(self):
        # data
        self.train: str = 'corpus/en_cn_test1.txt'
        self.valid: str = 'corpus/nist02_test.cn'
        self.valid_trg = ['corpus/nist02_test.en0',  'corpus/nist02_test.en1',
                          'corpus/nist02_test.en2', 'corpus/nist02_test.en3']
        self.valid_src = 'corpus/nist_c2e_02_src.utf8'
        self.valid_ref = 'corpus/nist_c2e_02_ref.norm'

        self.test: str = 'corpus/nist05_test.cn'
        self.test_trg = ['corpus/nist05_test.en0', 'corpus/nist05_test.en1',
                         'corpus/nist05_test.en2', 'corpus/nist05_test.en3']
        self.test_src = 'corpus/nist_c2e_05_src.utf8'
        self.test_ref = 'corpus/nist_c2e_05_ref.norm'
        self.checkpoint_dir = './training_checkpoints'

        # vocabulary
        self.path_to_vob_src = 'E:/Paii/repos/ver_1_2_3/vob.src'
        self.path_to_vob_trg = 'E:/Paii/repos/ver_1_2_3/vob.tgt'
        self.encoder_src = SentenceEncoder(self.path_to_vob_src)
        self.encoder_trg = SentenceEncoder(self.path_to_vob_trg)
        self.vocab_src_size = 30000
        self.vocab_trg_size = 30000

        # model
        self.enc_layers = 4
        self.rnn_type = 'LSTM'
        self.epochs = 1000 #6
        self.batch_size = 2 #64
        self.embedding_dim = 256 # 256
        self.hidden_units = 256 # 1024
        self.evaluate_frequency = 2 #frequency to compute bleu
        self.max_length_src = 100
        self.max_length_trg = 200
        self.GPU: int = '1'  # which_GPU_to_run: [0, 4), and -1 denote CPU.


    def verify(self):
        assert os.path.isfile(self.train)
        if self.valid is not None:
            assert os.path.isfile(self.valid)




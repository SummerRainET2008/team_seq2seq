from ver_1_2_3 import *
from ver_1_2_3.common.subword import sentence_encoder
import sentencepiece as spm

class SubtokenizerTest(tf.test.TestCase):
  line = "hello world, I am going to see you!"

  def _init_subtokenizer(self):
    temp_file = nlp.get_new_temporay_file()
    print(self.line, file=open(temp_file, "w"))

    train_param = f"--input={temp_file} " \
                  f"--pad_id={PAD_ID} " \
                  f"--unk_id={UNK_ID} " \
                  f"--bos_id={BOS_ID} " \
                  f"--eos_id={EOS_ID} " \
                  f"--model_prefix=_test_model " \
                  f"--vocab_size={20} " \
                  f"--model_type=bpe"
    spm.SentencePieceTrainer.Train(train_param)

    encoder = spm.SentencePieceProcessor()
    encoder.load("_test_model.model")

    return encoder

  def test_encode(self):
    encoder = self._init_subtokenizer()
    ids = encoder.encode_as_ids(self.line)
    out_line = encoder.decode_ids(ids)
    self.assertEqual(self.line, out_line)




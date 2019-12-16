# from common.subword.create_subwords import *
from common import *
import sentencepiece as spm

class SentenceEncoder(object):
  def __init__(self, model_prefix):
    self._encoder = spm.SentencePieceProcessor()
    self._encoder.load(f"{model_prefix}.model")

  def encode(self, line: str, add_eos=False):
    """Encodes a string into a list of int subtoken ids.
    line is after segmentation or toeknization.
    """
    # as <s> and </s> are special tokens in bpe.
    line = line.replace("<s>", "").replace("</s>", "")
    ids = self._encoder.encode_as_ids(line)
    if add_eos:
      ids.append(EOS_ID)

    return ids

  def decode(self, ids: list):
    ids = list(map(int, ids))
    return self._encoder.decode_ids(ids)


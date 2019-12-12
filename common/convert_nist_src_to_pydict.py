#!/usr/bin/env python3
#Author: Summer Xia

'''
This script converts nist src file into pydict file, and do segmentation is it
is Chinese, or tokenization, if it is English.
'''

from ver_1_2_3 import *
from pa_nlp.chinese import segment_sentence
from ver_1_2_3.common._tokenizer import norm_western_language

def _preprocess_file(in_file: str, out_file: str,
                     tag_name: str, is_tag_Chinese: bool, is_tag_English: bool):
  def process_ch(ch_sent: str):
    ch_sent = " ".join(segment_sentence("".join(ch_sent.split())))
    return ch_sent

  def process_other_languages(other_sent: str):
    return norm_western_language(other_sent)

  def get_src_sent():
    for ln in open(in_file):
      if "<seg" in ln:
        assert "</seg>" in ln
        content = ln[ln.find(">") + 1: ln.rfind("<")]
        yield content

  def get_data_record():
    for src_sent in get_src_sent():
      d = {}
      if is_tag_Chinese:
        d[tag_name] = process_ch(src_sent)

      elif is_tag_English:
        d[tag_name] = process_other_languages(src_sent)

      yield d

  with open(out_file, "w") as fou:
    for rd in get_data_record():
      print(rd, file=fou)

def main():
  parser = optparse.OptionParser(usage="cmd [optons]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--tag_name", default="ch", help="default 'ch'")
  parser.add_option("--is_tag_Chinese", action="store_true")
  parser.add_option("--is_tag_English", action="store_true")
  parser.add_option("--in_file")
  parser.add_option("--out_file")
  (options, args) = parser.parse_args()

  assert options.in_file is not None
  assert options.out_file is not None
  assert options.tag_name is not None

  tf.logging.set_verbosity(tf.logging.INFO)

  _preprocess_file(
    options.in_file,
    options.out_file,
    options.tag_name,
    options.is_tag_Chinese,
    options.is_tag_English,
  )

if __name__ == "__main__":
  main()


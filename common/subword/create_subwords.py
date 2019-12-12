#!/usr/bin/env python3
#Author: Summer Xia

from common import *
import sentencepiece as spm
# from ver_1_2_3.param import Param

def get_data(data_fields: list, files: list):
  for ln in nlp.next_line_from_files(files):
    d = eval(ln)
    for field in data_fields:
      ln = d[field].replace("<s>", "").replace("</s>", "").strip()
      yield ln

def main():
  parser = optparse.OptionParser(usage="cmd [optons]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  (options, args) = parser.parse_args()

  tf.logging.set_verbosity(tf.logging.INFO)
  params = Param.get_param_set()
  params.display()

  options = [
    (params.src_name, params.vob_src_model_prefix, params.vob_src_size),
    (params.tgt_name, params.vob_tgt_model_prefix, params.vob_tgt_size),
  ]

  for tag, model_prefix, vob_size in options:
    txt_file = nlp.get_new_temporay_file()

    try:
      tf.logging.info(f"output txt_file: {txt_file}")
      with open(txt_file, "w") as fou:
        for ln in get_data([tag], params.train_data):
          print(ln, file=fou)

      tf.logging.info(f"training...")
      train_param = f"--input={txt_file} " \
                    f"--pad_id={PAD_ID} " \
                    f"--unk_id={UNK_ID} " \
                    f"--bos_id={BOS_ID} " \
                    f"--eos_id={EOS_ID} " \
                    f"--model_prefix={model_prefix} " \
                    f"--vocab_size={vob_size} " \
                    f"--model_type=bpe"
      spm.SentencePieceTrainer.Train(train_param)
      tf.logging.info(f"training[{tag}] is done!")

    finally:
      nlp.execute_cmd(f"rm -v {txt_file}")

if __name__ == "__main__":
  main()


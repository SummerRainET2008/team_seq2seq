#!/usr/bin/env python3
#Author: Summer Xia

'''
This script deals with Chinese segmentation and English toknization at the
same time.

'''

from ver_1_2_3 import *
from pa_nlp.chinese import segment_sentence
from ver_1_2_3.common._tokenizer import norm_western_language

def _preprocess_file(in_file: str,
                     src_tag_name: str, is_src_tag_Chinese: bool,
                     tgt_tag_name: str, is_tgt_tag_Chinese: bool,
                     is_remove_head_tail_tag: bool,
                     lower_case: bool):
  def process_ch(ch_sent: str):
    if remove_head_tail_tag:
      begin_tag, end_tag = "<s>", "</s>"
      ch_sent = ch_sent.strip().replace(begin_tag, "").replace(end_tag, "")
    ch_sent = " ".join(segment_sentence("".join(ch_sent.split())))
    return ch_sent

  def process_other_languages(other_sent: str):
    return norm_western_language(other_sent)

  def remove_head_tail_tag(ln: str):
    begin_tag, end_tag = "<s>", "</s>"
    ln = ln.strip().replace(begin_tag, "").replace(end_tag, "")
    return ln

  def get_data_record():
    nlp.ensure_random_seed_for_one_time()

    for idx, ln in enumerate(nlp.next_line_from_file(in_file)):
      d = eval(ln)
      if "hash_id" not in d:
        d["hash_id"] = random.randint(0, 1 << 64)

      src_sent = d.get(src_tag_name, None)
      tgt_sent = d.get(tgt_tag_name, None)

      if src_sent is None or tgt_sent is None:
        tf.logging.warn(f"src or tgt is none: line[{idx + 1}]: '{ln}'")
        continue

      if lower_case:
        src_sent = src_sent.lower()
        tgt_sent = tgt_sent.lower()

      if is_remove_head_tail_tag:
        src_sent = remove_head_tail_tag(src_sent)
        tgt_sent = remove_head_tail_tag(tgt_sent)

      if is_src_tag_Chinese:
        d[src_tag_name] = process_ch(src_sent)
      else:
        d[src_tag_name] = process_other_languages(src_sent)

      if is_tgt_tag_Chinese:
        d[tgt_tag_name] = process_ch(tgt_sent)
      else:
        d[tgt_tag_name] = process_other_languages(tgt_sent)

      yield idx, d

  out_file = nlp.replace_file_name(in_file, ".pydict", ".seg.tok.pydict")
  with open(out_file, "w") as fou:
    for idx, rd in get_data_record():
      print(rd, file=fou)
      if (idx + 1) % 10000 == 0:
        tf.logging.info(f"{idx + 1} lines are processed.")

def main():
  parser = optparse.OptionParser(usage="cmd [optons]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--src_tag_name", default="ch", help="default 'ch'")
  parser.add_option("--is_src_tag_Chinese", action="store_true")
  parser.add_option("--tgt_tag_name", default="en", help="default 'en'")
  parser.add_option("--is_tgt_tag_Chinese", action="store_true")
  parser.add_option("--in_file")
  parser.add_option("--remove_head_tail_tag", action="store_true")
  parser.add_option("--lower_case", action="store_true")
  (options, args) = parser.parse_args()

  assert options.in_file is not None

  tf.logging.set_verbosity(tf.logging.INFO)

  _preprocess_file(
    options.in_file,
    options.src_tag_name, options.is_src_tag_Chinese,
    options.tgt_tag_name, options.is_tgt_tag_Chinese,
    options.remove_head_tail_tag,
    options.lower_case,
  )

if __name__ == "__main__":
  main()


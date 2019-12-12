#!/usr/bin/env python3
#Author: Summer Xia

from ver_1_2_3 import *

def create_vob(data_files: list, tag: str, out_vob_file: str):
  def get_sample():
    for ln in nlp.next_line_from_files(data_files):
      sample = eval(ln)
      yield sample[tag]

  word_stat = Counter()
  for idx, field in enumerate(get_sample()):
    if  idx > 0 and idx % 100_000 == 0:
      tf.logging.info(f"Processing {idx} lines")
    word_stat.update(field.split())

  vobs = word_stat.most_common(len(word_stat))
  nlp.write_pydict_file(vobs, out_vob_file)

def main():
  parser = optparse.OptionParser(usage="cmd [optons] data_file1 ...]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--tag", help="e.g., 'ch'", default="ch")
  parser.add_option(
    "--out_vob_file", default="vob.pydict", help="default vob.pydict"
  )
  (options, args) = parser.parse_args()

  tf.logging.set_verbosity(tf.logging.INFO)
  assert options.out_vob_file is not None

  create_vob(args, options.tag, options.out_vob_file)

if __name__ == "__main__":
  main()


#!/usr/bin/env python3
#Author: Summer Xia

from common import *
from common.bleu.postprocess.english import postprocess_en, deal_with_headline

reg_docid = re.compile('''docid="(.*?)"''', re.IGNORECASE)

def _extract_tags(ln):
  p1 = ln.find(">") + 1
  p2 = ln.rfind("<")

  return ln[: p1], ln[p1: p2], ln[p2: ]

def _get_doc_id(ln):
  rst = reg_docid.findall(ln)
  assert rst != []
  return rst[0]

def _get_next_hyp(nbest_file: str, buff={}):
  for d in nlp.pydict_file_read(nbest_file):
    nbest = d["nbest"]
    score, hyp = nbest[0] if nbest != [] else (0, "")
    yield hyp

def _convert_nbest_to_nist_format(nist_src: str, nbest_file: str):
  out_file = nlp.replace_file_name(nbest_file, ".pydict", ".tran")
  with open(nist_src) as fin_nist_src:
    with open(out_file, "w") as fou:
      hyp_iter = _get_next_hyp(nbest_file)

      doc_head = next(fin_nist_src).strip().replace("srcset", "tstset")
      print(doc_head, file=fou)

      is_head_line = False
      while True:
        try:
          ln = next(fin_nist_src).strip()
        except StopIteration:
          break

        ln = ln.strip().replace("</srcset>", "</tstset>")
        if "<doc" in ln or "<DOC" in ln or "<Doc" in ln:
          doc_name = _get_doc_id(ln)
          print(f'<doc docid="{doc_name}" sysid="chiero">', file=fou)

        elif ln.startswith("<seg"):
          assert "</seg>" in ln

          tag_head, _, tag_tail = _extract_tags(ln)
          hyp = next(hyp_iter)
          norm_hyp = postprocess_en(hyp)
          if is_head_line:
            norm_hyp = deal_with_headline(norm_hyp)

          print(f"{tag_head} {norm_hyp} {tag_tail}", file=fou)

        else:
          if "<hl>" in ln:
            is_head_line = True
          elif "</hl>" in ln:
            is_head_line = False

          print(ln, file=fou)

  return out_file

def _parse_bleu_score(bleu_file: str):
  # NIST score = 9.9508  BLEU score = 0.4180 for system "chiero"
  return float(re.findall(r"BLEU score = (.*?) ", open(bleu_file).read())[0])

def compute_bleu_score(nist_src: str, nist_ref: str, nbest_file: str):
  out_file = _convert_nbest_to_nist_format(nist_src, nbest_file)

  case_bleu_file = f"{out_file}.case.bleu"
  assert nlp.execute_cmd(
    f"./common/bleu/mteval-v11b.pl ",
    f"-r {nist_ref} -s {nist_src} -t {out_file} -c > {case_bleu_file}"
  ) == 0
  case_bleu = _parse_bleu_score(case_bleu_file)

  uncase_bleu_file = f"{out_file}.uncase.bleu"
  assert nlp.execute_cmd(
    f"./common/bleu/mteval-v11b.pl "
    f"-r {nist_ref} -s {nist_src} -t {out_file} > {uncase_bleu_file}"
  ) == 0
  uncase_bleu = _parse_bleu_score(uncase_bleu_file)

  return case_bleu, uncase_bleu

def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--in_file")
  parser.add_option("--nist_src")
  parser.add_option("--nist_ref")
  (options, args) = parser.parse_args()

  assert options.in_file is not None
  assert options.nist_src is not None

  tf.logging.set_verbosity(tf.logging.INFO)

  start_time = time.time()
  case_bleu, uncase_bleu = compute_bleu_score(
    options.nist_src, options.nist_ref, options.in_file
  )
  duration = time.time() - start_time

  tf.logging.info(
    f"{options.in_file}: "
    f"case-sensitive BLEU = {case_bleu:.4f}, "
    f"uncase-sensitive BLEU = {uncase_bleu:.4f}, "
    f"time: {duration:.2f} sec."
  )

if __name__ == "__main__":
  main()

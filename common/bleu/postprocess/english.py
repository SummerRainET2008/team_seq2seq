#!/usr/bin/env python3
#Author: Summer Xia

_head_line_words = set([
  "a",
  "after",
  "among",
  "an",
  "and",
  "as",
  "at",
  "before",
  "by",
  "for",
  "from",
  "in",
  "into",
  "of",
  "off",
  "on",
  "or",
  "out",
  "that",
  "the",
  "this",
  "to",
  "under",
  "up",
  "with",
  "within",
  "without",
])

def _upper_case_first_char(line: str):
  return line[0].upper() + line[1:]

def deal_with_headline(line: str):
  return " ".join(
    [w if w in _head_line_words else _upper_case_first_char(w)
     for w in line.split()]
  )

def _capitalize_intern_sentences(line: str):
  words = line.split()
  if words == []:
    return ""

  ret = [_upper_case_first_char(words[0])]
  next_word = iter(words[1:])

  for w in next_word:
    if ret[-1] in ".?!" or \
      len(ret) >= 2 and ret[-1] == '"' and ret[-2] in ":,":
      ret.append(_upper_case_first_char(w))

    else:
      ret.append(w)

  return " ".join(ret)

def _detokenize(line: str):
  replaces = [
    (" 's", "'s"),
    (" 're", "'re"),
    (" 't", "'t"),
    (" 'm", "'m"),
    (" 'll", "'ll"),
    (" 'd", "'d"),
    (" 've", "'ve"),
    ("s '", "s'"),
    (" n't", "n't"),
    (", ,", ","),
    (", .", "."),
    (". ,", ","),
  ]
  for old, new in replaces:
    line = line.replace(old, new)

  return line

def _drop_chinese_oov(line: str):
  #todo
  return line

def postprocess_en(line: str):
  line = line.replace("<s>", "").replace("</s>", "")
  line = line.replace("<unk>", "")

  line = _capitalize_intern_sentences(line)
  line = _detokenize(line)
  line = _drop_chinese_oov(line)

  return line.strip()




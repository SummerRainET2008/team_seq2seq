#!/usr/bin/env python3
#Author: Summer Xia

#todo: to support abbreviation dictionary.

from ver_1_2_3 import *

class _AbbrProtector:
  # I . O . U . --> I.O.U.
  # U . S . A --> U.S.A
  # the order is sensitive.
  regs = [
    re.compile(r"(([A-Z]\s*\.)+(A-Z))\b"),
    re.compile(r"(([A-Z]\s*\.)+)")
  ]

  def __init__(self):
    self._marks = {}

  def enter(self, line: str):
    for reg in _AbbrProtector.regs:
      for abb_raw in reg.findall(line):
        abb_raw = abb_raw[0]
        abb = "".join(abb_raw.split())
        abb_rep = f"abbr{abb.replace('.', '')}abbr"
        line = line.replace(abb_raw, abb_rep)
        self._marks[abb] = abb_rep

    return line

  def exit(self, line):
    for abb, abb_rep in self._marks.items():
      line = line.replace(abb_rep, abb)

    return line

_replace_punts = [
  ("，", " , "),
  ("？", " ? "),
  ("。", " . "),
  ("“", ''' " '''),
  ("”", ''' " '''),
  ("‘", """ ' """ ),
  ("’", """ ' """ ),
  (" ！", """ ! """ ),
  ("<skipped>", ""),
  ("&quot;", '"'),
  ("&amp;", "&"),
  ("&lt;", "<"),
  ("&gt;", ">"),
]

_replace_regs = [
  # tokenize punctuation
  (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/.,'])"), r" \1 "),

  # tokenize dash when preceded by a digit
  (re.compile(r"([0-9])(-)"), r"\1 \2 "),

  (re.compile(r"([a-zA-Z]+)\s+-\s+([a-zA-Z]+)"), r"\1-\2"),

  (re.compile(r"([0-9])\s*([\.,])\s*([0-9])"), r"\1\2\3"),

  (re.compile(r"([\w\d]+)'s"), r"\1 's"),

  # summer ' s --> summer 's
  (re.compile(r"\s+'\s+(s|re|t|m|ll|d|ve) "), r" '\1 "),

  # ' ' --> ""
  (re.compile(r"'\s+'"), "\""),

  (re.compile(r"o\s*'\s*clock"), "o'clock"),

  # one space only between words
  (re.compile(r"\s+"), " "),
]

def norm_western_language(line: str):
  abbr_protector = _AbbrProtector()
  line = abbr_protector.enter(line)

  for old_punt, new_punt in _replace_punts:
    line = line.replace(old_punt, new_punt)

  for reg, s in _replace_regs:
    line = reg.sub(s, line)

  line = abbr_protector.exit(line)

  return line.strip()


#!/usr/bin/env python3

from ver_1_2_3 import *
from ver_1_2_3.common._tokenizer import norm_western_language

class NormalizationTest(tf.test.TestCase):
  def test_normalize(self):
    samples = [
      (
        "Hammered down the right - field line . Justice!",
        "Hammered down the right-field line . Justice !",
      ),
      # (
      #   "1,2 , 3 ! Rise",
      #   "1, 2 , 3 ! Rise",
      # ),
      (
        "One against 500.",
        "One against 500 ."
      ),
      (
        "In five minutes for $49.95.",
        "In five minutes for $ 49.95 ."
      ),
      (
        "An hour. No ,50 minutes.",
        "An hour . No , 50 minutes .",
      ),
      (
        "I mean... You know, I'm hurt.",
        "I mean . . . You know , I 'm hurt .",
      ),
      (
        "One o'clock. Hugging the ground.",
        "One o'clock . Hugging the ground ."
      ),
      (
        "I am from U.S.",
        "I am from U.S.",
      ),
      (
        "I am from U.S.A, and how about you? ",
        "I am from U.S.A , and how about you ?"
      ),
      (
        "I am from U.S, and how about you? ",
        "I am from U.S , and how about you ?"
      ),
      (
        "$134,567.",
        "$ 134,567 .",
      ),
      (
        "A year with the U.N. Peacekeepers in Sudan.",
        "A year with the U.N. Peacekeepers in Sudan .",
      ),
      (
        "A year with the U.S. Peacekeepers in Sudan.",
        "A year with the U.S. Peacekeepers in Sudan .",
      ),
      (
        "A year with the U.S.A Peacekeepers in Sudan.",
        "A year with the U.S.A Peacekeepers in Sudan .",
      ),
      (
        "It's a signed I.O.U. From Thomas Edison.",
        "It 's a signed I.O.U. From Thomas Edison .",
      ),
      (
        "summer-rain 0-9",
        "summer-rain 0 - 9"
      )
    ]

    for raw_sample, exp_sample in samples:
      tokenized_sample = norm_western_language(raw_sample)
      self.assertEqual(
        tokenized_sample, exp_sample,
        # f"exp='{exp_sample}'\nout='{tokenized_sample}'"
      )

def main():
  tf.test.main()

if __name__ == "__main__":
  main()

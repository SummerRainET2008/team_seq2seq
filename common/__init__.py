from pa_nlp import *
import pa_nlp.common as nlp
from pa_nlp.tf_2x import nlp_tf
import six
import tensorflow as tf
import unicodedata

PAD     = "<PAD>"
PAD_ID  = 0          # This is important. Do NOT change.

UNK     = "<UNK"
UNK_ID  = 1

BOS     = "<s>"
BOS_ID  = 2

EOS     = "</s>"
EOS_ID  = 3

INF       =  1e7
NEG_INF   = -1e9
EPSILION  = 1e-8


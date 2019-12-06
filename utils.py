import sys
sys.path.append("E:/Paii")
from nmt_ver_1_1 import *
import tensorflow as tf


def max_length(tensor):
    return max(len(t) for t in tensor)

def convert_data(encoder_, sentence):
    encoded = [encoder_.encode(v) for v in sentence.split()]
    padded = [SOS] + [item for l in encoded for item in l] + [EOS]
    # padded = padded + [PAD] * max(0, max_length - len(sentence))
    return padded

def convert_data_0(encoder_, lang):
    # lang: a list of sentences
    # str to id with padding
    padded = []
    for x in lang:
        padded.append( convert_data(encoder_, x) )
        # padded is supposed to be like [[2180, 2180, 5007, 5008], [184, 184, 184, 184], ...]
        tensor = tf.keras.preprocessing.sequence.pad_sequences(padded, padding='post')
        # post: pad after each sequence
    assert len(tensor) == len(lang)
    return tensor


def generate_src_ref_list(src_file, trg_files):
    src_list = []
    with open(src_file, 'r', encoding='utf8') as f:
        for line in f:
            src = line.strip('\n')
            src_list.append(src)

    N = len(trg_files)
    ref_all = []
    for i in range(N):
        cur_ref = []
        with open(trg_files[i], 'r', encoding='utf8') as f:
            for line in f:
                trg = line.strip('\n')
                cur_ref.append(trg.split())
        ref_all.append(cur_ref)
    ref_list = [list(item) for item in zip(*ref_all)]

    return src_list, ref_list

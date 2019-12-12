import tensorflow as tf
import numpy as np
from ver_2 import *
from ver_2.param import Param
from pa_nlp.tf_2x import nlp_tf


# write tfrecord format train file
# data_file: original train file
def gen_train_data(data_file, tf_file, param):
  class Serializer:
    def __call__(self, sample):
      x, y = sample

      x = np.array(x, np.int32)
      y = np.array(y, np.int32)

      features = {
        "x": nlp_tf.tf_feature_bytes(x.tobytes()),
        "y": nlp_tf.tf_feature_bytes(y.tobytes())
      }

      example_proto = tf.train.Example(
        features=tf.train.Features(feature=features)
      )

      yield example_proto.SerializeToString()

  def get_file_record(data_file, param):
    with open(data_file, 'r', encoding='utf-8') as f:
      for line in f:
        en_cn = line.strip().split('\t')
        en = convert_data(
          param.encoder_en, en_cn[0], pad=True, max_length=param.max_length_trg
        )
        cn = convert_data(
          param.encoder_cn, en_cn[1], pad=True, max_length=param.max_length_src
        )
        yield cn, en # for sample in seg_samples seg_samples is a list

  nlp_tf.tfrecord_write(get_file_record(data_file, param), Serializer(), tf_file)

def gen_test_data(data_file, tf_file, param):
  class Serializer:
    def __call__(self, sample: list):
      sample = np.array(sample, np.int32)
      feature = {
          "x": nlp_tf.tf_feature_bytes(sample.tobytes())
        }
      example_proto = tf.train.Example(
          features=tf.train.Features(feature=feature)
        )

      yield example_proto.SerializeToString()

  def get_file_record(data_file, param):
    with open(data_file, 'r', encoding='utf-8') as f:
      for line in f:
        cn = convert_data(param.encoder_cn, line.strip(), pad=True, max_length=param.max_length_src)
        yield cn

  nlp_tf.tfrecord_write(get_file_record(data_file, param), Serializer(), tf_file)


def convert_data(encoder_, sentence, pad=False, max_length=None):
  encoded = [encoder_.encode(v) for v in sentence.split()]
  padded = [SOS] + [item for l in encoded for item in l] + [EOS]
  if pad == True:
    padded = padded + [PAD] * max(0, max_length - len(padded))
  return padded


if __name__ == "__main__":
  param = Param()
  gen_train_data('corpus/en_cn_test1.txt', param.train, param)
  gen_train_data('corpus/en_cn_test1.txt', param.vali_file, param)
  gen_train_data('corpus/nist02_test.cn', param.test_files[0], param)
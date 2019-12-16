import tensorflow as tf
import io
from ver_2_1 import *
from ver_2_1.param import Param
from pa_nlp.tf_2x import nlp_tf

# get dataset ftom tfrecord file
def get_batch_data(tf_file, epoch_num, batch_size, shuffle: bool, train = True):
  def parse_example(serialized_example):

    if train:
      data_fields = {
        "x": tf.io.FixedLenFeature((), tf.string, ""),
        "y": tf.io.FixedLenFeature((), tf.string, ""),
      }

      parsed_example = tf.io.parse_single_example(
        serialized_example, data_fields
      )

      x = tf.io.decode_raw(parsed_example["x"], tf.int32)
      y = tf.io.decode_raw(parsed_example["y"], tf.int32)

      return x, y

    else:
      data_fields = {
        "x": tf.io.FixedLenFeature((), tf.string, "")
      }

      parsed_example = tf.io.parse_single_example(
        serialized_example, data_fields
      )

      x = tf.io.decode_raw(parsed_example["x"], tf.int32)

      return x


  dataset = nlp_tf.tfrecord_read(
    file_pattern=tf_file,
    parse_example_func=parse_example,
    epoch_num=epoch_num,
    batch_size=batch_size,
    shuffle=shuffle,
    file_sequential=True
  )

  yield from dataset


if __name__ == '__main__':
  param = Param()
  dataset = get_batch_data(param.vali_tf, 1, 1, shuffle=False, train=False)
  for data in dataset:
    print(data)
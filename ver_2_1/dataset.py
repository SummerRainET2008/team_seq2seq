import tensorflow as tf
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

  data_read_iter = get_batch_data(param.train, param.epoch_num,
                                  param.batch_size, False)

  data_src, data_tgt = [], []
  for data in data_read_iter:
    src, tgt = data[2]
    data_src.extend(src.numpy().tolist())
    data_tgt.extend(tgt.numpy().tolist())

    print(data)
    print(len(data_src), len(data_tgt))

  pickle.dump([data_src, data_tgt], open("/tmp/input.data", "wb"))

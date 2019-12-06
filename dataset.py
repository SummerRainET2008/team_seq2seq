import tensorflow as tf
import io
import sys
sys.path.append("E:/Paii/")
from nmt_ver_1_1 import *
from utils import convert_data, convert_data_0, max_length

def gen_batch(data_file, encoder_src, encoder_trg, batch_sz, shuffle = None):
    def data_generator():
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                en_cn = line.strip().split('\t')
                en = convert_data(encoder_trg, en_cn[0])
                cn = convert_data(encoder_src, en_cn[1])
                yield en, cn
    data = tf.data.Dataset.from_generator(data_generator, (tf.int32, tf.int32), (tf.TensorShape([None]), tf.TensorShape([None])))
    data = data.padded_batch(batch_sz,
                             (tf.TensorShape([None]), tf.TensorShape([None])),
                             padding_values=(PAD, PAD),
                             drop_remainder=False)
    # data = data.prefetch(3)
    for batch_idx, batch in enumerate(data):
        yield batch_idx, batch


# original txt file to tf dataset including input language and target language
def to_tf_dataset(path, batch_sz, encoder_src, encoder_trg, num_examples=None):
    # path file format: en[tab]cn
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[w.strip() for w in l.split('\t')] for l in lines[:num_examples]]
    en, cn = zip(*word_pairs)
    src_tensor = convert_data_0(encoder_src, cn)
    trg_tensor = convert_data_0(encoder_trg, en)
    assert len(src_tensor) == len(trg_tensor)
    data_size = len(src_tensor)
    max_length_src = max_length(src_tensor)
    max_length_trg = max_length(trg_tensor)

    dataset = tf.data.Dataset.from_tensor_slices((src_tensor, trg_tensor)).shuffle(len(src_tensor))
    dataset = dataset.batch(batch_sz, drop_remainder=True)
    return dataset, data_size, max_length_src, max_length_trg


def gen_batch_0(inputs, targets, batch_sz, shuffle=False):
    # inputs: (seq_len, buffer_size)
    # Number of chunks
    buffer_size = len(inputs)
    steps_per_epoch = buffer_size // batch_sz

    for i in range(steps_per_epoch):
        batch_inputs = inputs[i*batch_sz, (i+1)*batch_sz]
        batch_targets = targets[i*batch_sz, (i+1)*batch_sz]
        yield batch_inputs, batch_targets
        # for batch_idx, (batch_inputs, batch_targets) in enumerate(gen_batch(inputs, targets, batch_sz)):
        # train_step(batch_inputs, batch_targets)

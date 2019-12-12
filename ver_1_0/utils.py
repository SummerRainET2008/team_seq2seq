import tensorflow as tf


def max_length(tensor):
    return max(len(t) for t in tensor)


def convert_data(encoder_, lang, sos=2, eos=3):
    # str to id with padding
    padded = []
    for x in lang:
        encoded = [encoder_.encode(v) for v in x.split()]
        padded.append(
            ([] if sos is None else [sos]) +
            [item for l in encoded for item in l] +
            ([] if eos is None else [eos]))
        # padded is supposed to be like [[2180, 2180, 5007, 5008], [184, 184, 184, 184], ...]
        tensor = tf.keras.preprocessing.sequence.pad_sequences(padded, padding='post')
        # post: pad after each sequence
        # padded[-1] = padded[-1] + [pad] * max(0, max_len - len(x))
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

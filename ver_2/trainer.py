import tensorflow as tf
import numpy as np
import os
import time
from pa_nlp.nlp import Logger
from pa_nlp.tf_2x import *
from pa_nlp.tf_2x.estimator.train import TrainerBase
from pa_nlp.tf_2x import nlp_tf

from ver_2.param import Param
from ver_2._model import Model
from ver_2.dataset import get_batch_data

from ver_2 import *
from common.bleu.compute_bleu_by_nist_format import compute_bleu_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


class Trainer(TrainerBase):
  def __init__(self, param):
    model = Model(param)

    data_read_iter = get_batch_data(param.train, param.epoch_num,
                                    param.batch_size, False)
    super(Trainer, self).__init__(param, model, data_read_iter)

  def _train_one_batch(self, src, trg):
    with tf.GradientTape() as tape:
      loss = self._model(src, trg)
    batch_loss = (loss / int(trg.shape[1]))
    self._apply_optimizer(tape, loss)
    return batch_loss

  def predict(self, src):  # src is in get_batch_data format
    predictions = self._model.predict(src, self._param.max_length_trg)
    result = self._param.encoder_en.decode(predictions)
    # result += encoder_trg.decode([predicted_id]) + ' '
    return result.rstrip()  # string

  def evaluate_file(self, data_file):
    # data_file: valid file in tfrecord format
    start_time = time.time()
    error = 0
    for _, idx, batch in get_batch_data(data_file, 1, 1, shuffle=False):
      src, trg = batch
      error += self._model(src, trg) / int(trg.shape[1])

    error = error / (idx + 1)
    total_time = time.time() - start_time
    avg_time = total_time / (idx + 1)
    Logger.info(
      f"eval[{self._checkpoint.run_samples.numpy()}]: "
      f"file={data_file} error={error} "
      f"total_time={total_time:.4f} secs avg_time={avg_time:.4f} sec/sample "
    )

    dirname = data_file.split('/')[-1].split('.')[0]
    bleu_src = f"corpus/{dirname}.src"
    bleu_ref = f"corpus/{dirname}.ref"
    if os.path.exists(bleu_src) and os.path.exists(bleu_ref):
      all_to_export = []
      for _, _, batch in get_batch_data(data_file, 1, 1, shuffle=False):
        to_export = dict()
        src, trg = batch
        result = self.predict(src)
        to_export['ch'] = self._param.encoder_cn.decode(
          src.numpy()[0].tolist())
        to_export['nbest'] = []
        to_export['nbest'].append((None, result))
        all_to_export.append(to_export)

      if not os.path.exists(f"{self._param.path_bleu}/{dirname}"):
        os.mkdir(f"{self._param.path_bleu}/{dirname}")
      path_to_pydict = f"{self._param.path_bleu}/{dirname}/" \
                       f"{self._checkpoint.run_samples.numpy()}.nbest.pydict"
      with open(path_to_pydict, 'w', encoding='utf8') as f:
        f.writelines(f'{item}\n' for item in all_to_export)
      print('{} saved!'.format(path_to_pydict))
      try:
        bleu, _ = compute_bleu_score(bleu_src, bleu_ref, path_to_pydict)
        print('BLEU for run samples {} is {}'.format(
          self._checkpoint.run_samples.numpy(), bleu))
      except AssertionError:
        print('BLEU score compute error!')

    return error


if __name__ == '__main__':
  param = Param()
  # param.verify()
  trainer = Trainer(param)
  trainer.train()

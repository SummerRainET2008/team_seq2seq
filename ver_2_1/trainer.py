from common.bleu.compute_bleu_by_nist_format import compute_bleu_score
from pa_nlp.nlp import Logger
from pa_nlp.tf_2x import *
from pa_nlp.tf_2x.estimator.train import TrainerBase
from ver_2_1._model import Model
from ver_2_1.dataset import get_batch_data
from ver_2_1.param import Param

class Trainer(TrainerBase):
  def __init__(self, param):
    model = Model(param)

    data_read_iter = get_batch_data(param.train, param.epoch_num,
                                    param.batch_size, False)
    super(Trainer, self).__init__(param, model, data_read_iter)

  @tf.function(
    input_signature=(
      tf.TensorSpec(shape=[None, 50], dtype=tf.int32),
      tf.TensorSpec(shape=[None, 50], dtype=tf.int32)
    )
  )
  def _train_one_batch(self, src, trg):
    with tf.GradientTape() as tape:
      loss = self._model(src, trg)
    batch_loss = (loss / int(trg.shape[1]))
    self._apply_optimizer(tape, loss)
    return batch_loss

  @tf.function(
    input_signature=(
      tf.TensorSpec(shape=[None, 50], dtype=tf.int32),
      tf.TensorSpec(shape=[], dtype=tf.int32)
    )
  )
  def predict(self, src, max_length_trg):
    predictions = self._model.predict(src, max_length_trg)
    return predictions


  def evaluate_file(self, data_file):
    # data_file: valid file in tfrecord format
    # compute minus bleu
    start_time = time.time()

    dirname = data_file.split('/')[-1].split('.')[0]
    bleu_src = f"corpus/{dirname}.src"
    bleu_ref = f"corpus/{dirname}.ref"
    if os.path.exists(bleu_src) and os.path.exists(bleu_ref):
      all_to_export = []
      for _, idx, batch in get_batch_data(data_file, 1, 1, shuffle=False):
        to_export = dict()
        src, trg = batch
        predictions = self.predict(src, self._param.max_length_trg)
        result = self._param.encoder_en.decode(list(predictions.numpy()))
        to_export['ch'] = self._param.encoder_cn.decode(
          src.numpy()[0].tolist())
        to_export['nbest'] = []
        to_export['nbest'].append((None, result.strip()))
        all_to_export.append(to_export)

      total_time = time.time() - start_time
      avg_time = total_time / (idx + 1)

      if not os.path.exists(f"{self._param.path_bleu}/{dirname}"):
        os.mkdir(f"{self._param.path_bleu}/{dirname}")
      path_to_pydict = f"{self._param.path_bleu}/{dirname}/" \
                       f"{self._checkpoint.run_sample_num.numpy()}." \
                       f"nbest.pydict"
      with open(path_to_pydict, 'w', encoding='utf8') as f:
        f.writelines(f'{item}\n' for item in all_to_export)
      Logger.info('{} saved!'.format(path_to_pydict))
      try:
        bleu, _ = compute_bleu_score(bleu_src, bleu_ref, path_to_pydict)
        Logger.info(
          f"eval[{self._checkpoint.run_sample_num.numpy()}]: "
          f"file={data_file} bleu={bleu} "
          f"total_time={total_time:.4f} secs "
          f"avg_time={avg_time:.4f} sec/sample "
        )
      except AssertionError:
        Logger.info('BLEU score compute error: Translations are empty!')
        bleu = 0

      return -bleu

    else:
      Logger.info('No reference file to compute BLEU')
      return 0


def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--gpu", default="-1", help="default=-1")
  parser.add_option("--debug_level", type=int, default=1)
  (options, args) = parser.parse_args()

  os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu
  Logger.set_level(options.debug_level)

  nlp.display_server_info()
  Logger.info(f"GPU: {options.gpu}")

  param = Param()
  param.verify()
  trainer = Trainer(param)
  trainer.train()

if __name__ == '__main__':
  main()


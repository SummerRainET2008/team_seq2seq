from ver_2 import *
import tensorflow as tf
import io
from ver_2.param import Param
from ver_2.trainer import Trainer
from ver_2.make_features import gen_test_data


class Predictor(object):
  def __init__(self, param):
    self.param = param
    gen_test_data(param.test_files, param.test_tf, param)

  def load_model(self):
    Trainer.load_model()

  def evaluate_file(self):
    self.load_model()
    return Trainer.evaluate_file(self.param.test_tf, 0, 0)


if __name__ == "__main__":
  param = Param()
  predictor = Predictor(param)
  predictor.evaluate()
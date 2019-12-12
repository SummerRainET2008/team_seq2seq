import sys
sys.path.append("E:/Paii")
from nmt_with_attention_tf2 import *

from param import Param
from predictor import Predictor

param = Param()
predictor = Predictor(param)
predictor.load_model(param)
predictor.evaluate()
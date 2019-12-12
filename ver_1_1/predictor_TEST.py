import sys
sys.path.append("E:/Paii")
from nmt_ver_1_1 import *

from param import Param
from predictor import Predictor

param = Param()
predictor = Predictor(param)
predictor.load_model(param)
predictor.evaluate()
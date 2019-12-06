import sys
sys.path.append("E:/Paii")
from nmt_ver_1_1 import *

from param import Param
from trainer import Trainer

param = Param()
trainer = Trainer(param)
trainer.train(param)



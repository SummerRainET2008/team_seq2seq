import sys
sys.path.append("E:/Paii")
from nmt_with_attention_tf2 import *

from param import Param
from trainer import Trainer

param = Param()
trainer = Trainer(param)
trainer.train(param)



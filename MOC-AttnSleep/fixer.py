import argparse
import collections
import numpy as np

from data_loader.data_loaders import *
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils.util import *

import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import math
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

'''
outp = r"C:\Users\athul\myfiles\Research\Sleep Stage\AttnSleep\saved\Exp1\23_07_2023_20_47_17_fold1\outs_1.npy"
trg = r"C:\Users\athul\myfiles\Research\Sleep Stage\AttnSleep\saved\Exp1\23_07_2023_20_47_17_fold1\trgs_1.npy"
y_pred  = np.load(outp)
y_true  = np.load(trg)
target_names = ["W","N1","N2","N3","REM"]
print(y_true[:100])
print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=target_names))
print(confusion_matrix(y_true, y_pred))
'''
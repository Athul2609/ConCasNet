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
import pickle

import torch
import torch.nn as nn

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def one_fold_cross_val(config, fold_id,folds_data,order,level):
    # you can do this becuse of the _getitem_ method in your parse_config
    batch_size = config["data_loader"]["args"]["batch_size"]

    logger = config.get_logger('train')

    # build model architecture, initialize weights, then print to console
    model = config.init_obj('arch', module_arch) #returns attnsleep object or whatever is put in type in arch
    model.apply(weights_init_normal)

    # based on the level set for logger this will get logged
    logger.info(model)

    # get function handles of loss and metrics

    #creating loss function objection and a list of metric objects using  getattr method
    criterion = getattr(module_loss, config['loss']) #weightedcrossentropy
    metrics = [getattr(module_metric, met) for met in config['metrics']] #accuracy

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters()) #finding trainable params

    optimizer = config.init_obj('optimizer', torch.optim, trainable_params) #initialising adam

    data_loader, valid_data_loader, full_test_loader,full_train_loader,data_count = data_generator_np(folds_data[fold_id][0],
                                                                   folds_data[fold_id][1], batch_size, order, level) #check is shuffle
    weights_for_each_class = calc_class_weight(data_count,order,level)
    print(f"Weights {weights_for_each_class}")
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      fold_id=fold_id,
                      valid_data_loader=valid_data_loader,
                      full_test_loader=full_test_loader,
                      full_train_loader=full_train_loader,
                      class_weights=weights_for_each_class)

    trainer.train()

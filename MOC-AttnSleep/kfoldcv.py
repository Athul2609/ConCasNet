import argparse
import collections
import numpy as np

from parse_config import ConfigParser
from utils.util import *
from train_Kfold_CV import one_fold_cross_val
from model_fusion import constructer,rearranger

import pickle

import sys

#Experiment name
NAME="attempt_1"
EPOCH = 1

def set_expn_name(name):
    file_path="./config.json"
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    data["name"] = name

    # Write the modified data back to the file
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    # Print the updated name
    print("Updated Experiment Name:", data["name"] )

def set_savdir_name(name):
    file_path="./config.json"
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    data["trainer"]["save_dir"] = name

    # Write the modified data back to the file
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    # Print the updated name
    print("Updated Experiment Save dir:", data["trainer"]["save_dir"])


def kfold(args,level):
    options=[]
    args2 = args.parse_args()
    for fold_id in range(20):
        # if fold_id < 13:
        #     continue
        print(f"Fold number {fold_id+1}")
    # config is a Configparser object
        config = ConfigParser.from_args(args, fold_id, options)
        if "SHHS" in args2.np_data_dir:
            folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
        else:
            folds_data = load_folds_data(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

        f_name = "data_" + str(fold_id)
        with open(config.save_dir / f_name, 'wb') as file:
            pickle.dump(folds_data[fold_id], file)
        order=[int(i) for i in args2.order.split()]
        one_fold_cross_val(config, fold_id,folds_data,order,level)
        # sys.exit()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-o', '--order', type=str, help='order')
    args.add_argument('-da', '--np_data_dir', type=str,
                      help='Directory containing numpy files')
    
    args2 = args.parse_args()
    savdir = f'C:/Users/athul/myfiles/Work/Research/Sleep Stage/SHHS Cascading Order Analysis Result Construction/saved_SHHS/{(args2.order).replace(" ","_")}'    
    set_savdir_name(savdir)
    # for i in range(4):
    #     set_expn_name("model"+str(i+1))
    #     kfold(args,i)

    if "SHHS" in args2.np_data_dir:
        data_paths = load_folds_data_shhs(args2.np_data_dir, 20)
    else:
        data_paths = load_folds_data(args2.np_data_dir, 20)
    for i in range(20):
        data_paths[i][0]=sorted(data_paths[i][0],key=lambda x:x.split('\\')[-1][8:12])
    lst=data_paths[1][0].copy()
    lst.extend(data_paths[1][1])
    lst=sorted(lst,key=lambda x:x.split('\\')[-1][8:12]) #use [8:12] for shhs and [3:6] for sleep-edf
    rearranger(lst)
    order=[int(i) for i in args2.order.split()]
    constructer(order)

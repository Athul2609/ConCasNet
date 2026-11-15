"""Train the four binary classifiers of the ConCasNet cascade with 20-fold
cross-validation, then fuse their outputs into 5-class predictions.

    python kfoldcv.py --order "3 0 2 4 1" --np_data_dir <folder of .npz files> [--save_dir saved]

--order       cascade order as label indices (0 Awake, 1 N1, 2 N2, 3 N3, 4 REM).
              "3 0 2 4 1" = N3 > Awake > N2 > REM > N1, the order used in the paper.
--np_data_dir the dataset is recognised from this path (see utils/util.py):
              contains "SHHS" -> SHHS, contains "78" -> Sleep-EDF-78, otherwise Sleep-EDF-20.
--save_dir    results go to <save_dir>/<order>/model1..model4 and <save_dir>/<order>/Combined.

For SHHS, also edit the two lines marked DATASET SWITCH in model/model.py.
"""
import argparse
import os

from parse_config import ConfigParser
from utils.util import *
from train_Kfold_CV import one_fold_cross_val
from model_fusion import constructer,rearranger

import pickle


def recording_key(path, np_data_dir):
    """Sort key for recording files: the subject/night digits of the file name.
    SHHS: shhs1-2xxxxx.npz -> characters 8:12.  Sleep-EDF: SC4ssNE0.npz -> characters 3:6."""
    name = os.path.basename(path)
    return name[8:12] if "SHHS" in np_data_dir else name[3:6]


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
    args.add_argument('-s', '--save_dir', default="saved", type=str,
                      help='Root folder for results (default: saved)')

    args2 = args.parse_args()
    savdir = os.path.join(args2.save_dir, args2.order.replace(" ", "_"))
    set_savdir_name(savdir)

    # 1) train the four classifiers (cascade levels 0-3), 20 folds each
    for i in range(4):
        set_expn_name("model"+str(i+1))
        kfold(args,i)

    # 2) put every model's training-set outputs in the same recording order
    if "SHHS" in args2.np_data_dir:
        data_paths = load_folds_data_shhs(args2.np_data_dir, 20)
    else:
        data_paths = load_folds_data(args2.np_data_dir, 20)
    lst=data_paths[1][0].copy()
    lst.extend(data_paths[1][1])
    lst=sorted(lst,key=lambda x: recording_key(x, args2.np_data_dir))
    rearranger(lst)

    # 3) fuse the four binary outputs into 5-class predictions (<save_dir>/<order>/Combined)
    order=[int(i) for i in args2.order.split()]
    constructer(order)

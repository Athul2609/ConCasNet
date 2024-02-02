import argparse
import os
import numpy as np
import pickle
from matrixes import transition_matrix,emission_matrix,breaker
from hmmlearn import hmm
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report,cohen_kappa_score,confusion_matrix
from util import calc_metrics
import pandas as pd

def len_finder(array):
    train=array[0]
    val=array[1]
    train_len=[]
    noz=0
    train_y=[]
    for i in train:
        y=np.load(i)["y"]
        train_len.append(len(y))
        if y[0]!=0 : noz+=1
        train_y.append(y)
    val_len=[]
    for i in val:
        y=len(np.load(i)["y"])
        val_len.append(y)
    return train_len,val_len,noz,train_y

def load_data_and_arrays(result_directory):
    file_prefixes = ["outs_train", "trgs_train", "outs_val", "trgs_val"]
    loaded_arrays = {}
    for prefix in file_prefixes:
        for root, dirs, files in os.walk(result_directory):
            for file in files:
                if prefix in file:
                    loaded_arrays[prefix]=result_directory+"\\"+file
    
    return loaded_arrays


def main(result_directory):
    loaded_arrays=load_data_and_arrays(result_directory)
    tm=transition_matrix([np.load(loaded_arrays["trgs_train"])])
    em=emission_matrix(loaded_arrays["outs_train"],loaded_arrays["trgs_train"])

    model = hmm.CategoricalHMM(n_components=5) 

    model.startprob_ = np.array([1.0, 0.0,0.0,0.0,0.0]) 
    model.transmat_ = tm
    model.emissionprob_ = em

    hidden_sequence = model.predict((np.load(loaded_arrays["outs_val"])).reshape(-1, 1).astype(np.int32))
    outs_name = "hmm_val"
    np.save(os.path.join(result_directory,outs_name), hidden_sequence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy files from a result directory to a saved directory.")
    parser.add_argument("result_directory", help="Path to the result directory")
    
    args = parser.parse_args()
    for root, directories, files in os.walk(args.result_directory):
        for directory in directories:
            inp_dir=os.path.join(root, directory)
            main(inp_dir)

    calc_metrics(args.result_directory)

# python hmm_input.py /path/to/result_directory
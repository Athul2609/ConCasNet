import argparse
import os
import numpy as np
import pickle
from matrixes import transition_matrix,emission_matrix,breaker,breaker_observed
from hmmlearn import hmm
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report,cohen_kappa_score
from util import calc_metrics


def load_data_and_arrays(result_directory):

    # Load .npy files as NumPy arrays with matching prefixes
    file_prefixes = ["outs_train", "trgs_train", "outs_val", "trgs_val","prob_val","prob_train"]
    loaded_arrays = {}
    for prefix in file_prefixes:
        for root, dirs, files in os.walk(result_directory):
            for file in files:
                if prefix in file and "hmm" not in file:
                    loaded_arrays[prefix]=os.path.join(result_directory,file)
    
    return loaded_arrays


def main(result_directory,weight,mode="val"):
    loaded_arrays=load_data_and_arrays(result_directory)
    # print(f"The paths are-\n {loaded_arrays}")


    tm=transition_matrix([np.load(loaded_arrays["trgs_train"])])
    em=emission_matrix(loaded_arrays["outs_train"],loaded_arrays["trgs_train"],loaded_arrays["prob_train"],weight)

    model = hmm.CategoricalHMM(n_components=5) 

    model.startprob_ = np.array([1.0, 0.0,0.0,0.0,0.0]) 
    model.transmat_ = tm
    model.emissionprob_ = em

    if mode=="train":
        observed_sequence=breaker_observed(loaded_arrays["outs_train"],loaded_arrays["prob_train"],weight)
        actual_labels=np.load(loaded_arrays["trgs_train"])
        n_trials = 1
        hidden_sequence = model.predict(observed_sequence.reshape(-1, 1).astype(np.int32))
        outs_name = f"outs_{mode}_hmm_weighted"
        np.save(os.path.join(result_directory,outs_name), hidden_sequence)
        return accuracy_score(actual_labels,hidden_sequence)
    # print("-------------validation---------------")
    else:
        observed_sequence=breaker_observed(loaded_arrays["outs_val"],loaded_arrays["prob_val"],weight)
        actual_labels=np.load(loaded_arrays["trgs_val"])
        n_trials = 1
        hidden_sequence = model.predict(observed_sequence.reshape(-1, 1).astype(np.int32))
        outs_name = f"outs_{mode}_hmm_weighted"
        np.save(os.path.join(result_directory,outs_name), hidden_sequence)
        return accuracy_score(actual_labels,hidden_sequence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy files from a result directory to a saved directory.")
    parser.add_argument("result_directory", help="Path to the result directory")
    parser.add_argument("weight", help="seperating probability")
    parser.add_argument("compute",default="no")
    parser.add_argument("mode",default="val")
    
    args = parser.parse_args()
    if(args.compute=="yes"):
        max_acc=0
        best_weights=[0.50,0.50,0.50,0.50,0.50]
        for i in range(5):
            for k in range(50,100,5):
                j=best_weights.copy()
                j=[float(g) for g in j]
                j=np.array(j)
                k=k/100
                j[i]=k
                for root, directories, files in os.walk(args.result_directory):
                    for directory in directories:
                        inp_dir=os.path.join(root, directory)
                        main(inp_dir,j,mode=args.mode)
                temp=calc_metrics(args.result_directory,j,args.mode)
                if(temp>max_acc):
                    print(f"Updating weight to {j}")
                    max_acc=temp
                    best_weights=j
                best_n=j[i]
            for l in range(-5,5,1):
                j=best_weights.copy()
                j=[float(g) for g in j]
                j=np.array(j)
                l=l/100
                j[i]=best_n+l
                for root, directories, files in os.walk(args.result_directory):
                    for directory in directories:
                        inp_dir=os.path.join(root, directory)
                        main(inp_dir,j,mode=args.mode)
                temp=calc_metrics(args.result_directory,j,args.mode)
                if(temp>max_acc):
                    print(f"Updating weight to {j}")
                    max_acc=temp
                    best_weights=j
        for root, directories, files in os.walk(args.result_directory):
            for directory in directories:
                inp_dir=os.path.join(root, directory)
                main(inp_dir,best_weights,"val")
        temp=calc_metrics(args.result_directory,best_weights,"val")
        print(f"Current accuracy : {temp}")
        print(f"Weights: {best_weights}")
        # accs.append(calc_metrics(args.result_directory,j))
    else:
        j=[float(i) for i in args.weight.split()]
        for root, directories, files in os.walk(args.result_directory):
            for directory in directories:
                inp_dir=os.path.join(root, directory)
                main(inp_dir,j,args.mode)
        temp=calc_metrics(args.result_directory,j,args.mode)
        print(f"accuracy {temp}")
# python hmm_input.py /path/to/result_directory
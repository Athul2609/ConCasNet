"""Utilities for fusing and analysing model outputs.

This module helps locate saved prediction files produced by
multiple models, merge their outputs according to various orders,
compute combined performance metrics and save results. It is used
in the ConCasNet sleep‑stage classification experiments to build
ensemble outputs over cross‑validation folds.

Functions
---------
get_final_lists : locate latest output, target and probability files
getFusionOutputPath : aggregate paths for all models/folds
merger : combine output arrays according to a specified order
prob_merger : merge probability arrays conforming to an order
rearranger : reorder and save intermediate files for training outputs
get_save_dir : create directory for combined results
constructer : orchestrate full fusion workflow and write reports
"""

import os
import json
import numpy as np
from sklearn.metrics import accuracy_score,cohen_kappa_score,classification_report,confusion_matrix
import pandas as pd
import torch
from result_formatter import reconstruct,order_fixer_pt2

#Experiment name
with open('./config.json', "r") as json_file:
        data = json.load(json_file)
NAME=data["name"]
EPOCH =data["trainer"]["epochs"]
SAVED=data["trainer"]["save_dir"]
model_prefix=["model1","model2","model3","model4"]

def get_final_lists(folder_path,pref):
    """Find the latest set of output/trg/prob files in a folder.

    Parameters
    ----------
    folder_path : str
        Path to the directory containing model2 output files from a single
        training run or fold.
    pref : str
        Prefix used when constructing training filenames (usually 'n' for
        normalized outputs).

    Returns
    -------
    list
        Six-element list with the names of the validation output file,
        its corresponding target file, probability file, and the analogous
        three training filenames derived from the validation name.

    Notes
    -----
    The function scans for files containing "model2_outs" but not "prob".
    If multiple candidates exist it chooses the one with the highest
    numerical suffix. The returned training names are constructed by
    replacing portions of the validation filename with the provided prefix.
    """
    output_file=None
    for file in os.listdir(os.path.abspath(folder_path)):
        if "model2_outs" in file and "prob" not in file:
            if output_file==None:
                output_file=file
            else:
                try:
                    if int(output_file.split("_")[-1].split(".")[0])<int(file.split("_")[-1].split(".")[0]):
                        output_file=file
                except ValueError as e:
                    print(output_file)
                    print(file)
                    print(str(e))
    if output_file!=None:
        trgs_file=output_file.replace("outs","trgs")
        prob_file=output_file.replace("outs","outs_prob")
        train_output_file=output_file.replace("model2_outsv",pref+"outs_train_")
        train_trgs_file=train_output_file.replace(pref+"outs",pref+"trgs")
        train_prob_file=train_output_file.replace(pref+"outs",pref+"outs_prob")
        return [output_file,trgs_file,prob_file,train_output_file,train_trgs_file,train_prob_file]
    else:
        folder_abs = os.path.abspath(folder_path)
        files_list = os.listdir(folder_abs)
        print(f"No output files found in {folder_abs}. Looking for files containing 'model2_outs' but not 'prob'. Available files: {files_list}")
        import sys
        sys.exit(1)

def getFusionOutputPath(pref='n'):
    """Gather model paths for all cross‑validation folds for each model

    Parameters
    ----------
    pref : str, optional Defaults to 'n'.

    Returns
    -------
    dict
        A mapping keyed by fold subfolder name (e.g. 'fold0') each containing
        lists of paths for train/validation outputs, probabilities and
        targets for each of the 4 models.
    
    """
    model_output_paths={}
    for fold_no in range(20):
        str_to_match = f"_fold{fold_no}"
        model_output_paths[str_to_match]={"train_outputs":[],"val_outputs":[],"train_prob":[],"val_prob":[],"train_trgs":["hi"],"val_trgs":["hi"]}
        for name in model_prefix:
            folder_path=os.path.abspath(os.path.join(SAVED,name))
            for files in os.listdir(folder_path):
                if os.path.isdir(os.path.abspath(os.path.join(folder_path,files))) and files.endswith(f"_fold{fold_no}"):
                    output_file,trgs_file,prob_file,train_output_file,train_trgs_file,train_prob_file=get_final_lists(os.path.abspath(os.path.join(folder_path,files)),pref)
                    model_output_paths[str_to_match]["val_outputs"].append(os.path.join(os.path.abspath(os.path.join(folder_path,files)),output_file))
                    model_output_paths[str_to_match]["train_outputs"].append(os.path.join(os.path.abspath(os.path.join(folder_path,files)),train_output_file))
                    model_output_paths[str_to_match]["train_prob"].append(os.path.join(os.path.abspath(os.path.join(folder_path,files)),train_prob_file))
                    model_output_paths[str_to_match]["val_prob"].append(os.path.join(os.path.abspath(os.path.join(folder_path,files)),prob_file))                    
                    model_output_paths[str_to_match]["train_trgs"][0]=(os.path.join(os.path.abspath(os.path.join(folder_path,files)),train_trgs_file))
                    model_output_paths[str_to_match]["val_trgs"][0]=(os.path.join(os.path.abspath(os.path.join(folder_path,files)),trgs_file))
                    break
        # print(f"Fold {str_to_match}: val_prob paths: {model_output_paths[str_to_match]['val_prob']}")
    return model_output_paths

def merger(arr,order,mode):
    """Merge discrete outputs from multiple models according to ordering.

    This function takes the dictionary produced by :func:`getFusionOutputPath`
    for a single fold and then loads the first two output arrays for the
    specified mode ('train' or 'val'). Outputs are recoded according to
    the provided ordering list. A series of fallback rules using additional
    temporary files is applied when the primary output value equals 1 or
    -1.

    Parameters
    ----------
    arr : dict
        Fold-specific mapping with keys like 'train_outputs' or 'val_trgs'
        containing lists of file paths.
    order : list of int
        The permutation applied to class indices during fusion.
    mode : {'train','val'}
        Determines which set of paths to use from ``arr``.

    Returns
    -------
    tuple
        ``(outputs, trgs)`` where both are numpy arrays of the merged
        predictions and the corresponding ground truth labels.
    """
        # arrt=arr.copy()
        # arrt=[single_path.replace("outs","trgs") for single_path in arrt] 
    
    trgs_label=f"{mode}_trgs"
    trgs_output=f"{mode}_outputs"
    trgs=arr[trgs_label][0]
    trgs=np.load(trgs)
    outputs=np.load(arr[trgs_output][0])
    temp=arr[trgs_output][1]
    # count=0
    temp=np.load(temp)
    for idx,i in enumerate(outputs):
        if i == 1:
            outputs[idx]=order[0]
            # count+=1
        else:
            try:
                if(temp[idx]==1):
                    outputs[idx]=order[1]
                else:
                    outputs[idx]=-1
            except IndexError as e:
                print(e)
                print(f"the length of Temp is {len(temp)}")
                print(f"The length of the outputs is {len(outputs)}")
                print(f"The index we are at {idx}")
                print(f"The index in temp is {idx}")
    # temp=arr[trgs_output][2]
    # temp=np.load(temp)
    # for idx,i in enumerate(outputs):
    #     if i == 2:
    #         if(temp[idx]==0):
    #             outputs[idx]=-1
    
    temp=arr[trgs_output][2]
    temp=np.load(temp)
    for idx,i in enumerate(outputs):
        if i == -1:
            if(temp[idx]==1):
                outputs[idx]=order[2]
    temp=arr[trgs_output][3]
    temp=np.load(temp)
    for idx,i in enumerate(outputs):
        if i==-1:
            if(temp[idx]==1):
                outputs[idx]=order[3]
            else:
                outputs[idx]=order[4]
    # print(accuracy_score(outputs,trgs))
    # print(outputs)
    # print(outputs.shape)
    return outputs,trgs

def prob_merger(arr,order,mode):
    """Combine probability arrays from several models into aligned vectors.

    For each time step the function collects the probability of the positive
    class from each model in the order specified. The ordering controls which
    model's probability occupies each slot in the resulting list.

    Parameters
    ----------
    arr : dict
        Mapping like that returned by :func:`getFusionOutputPath` for a fold.
    order : list of int
        Permutation used to reorder the models' contributions.
    mode : str
        Either ``'train_prob'`` or ``'val_prob'`` indicating which list to use
        inside ``arr``.

    Returns
    -------
    list of list
        Each element is a probability vector of length equal to the number of
        models (4 in this implementation) representing the fused probabilities
        for a single example.
    """
    all_probs=[]
    for i in arr[mode]:
        all_probs.append(np.load(i))
    probs=[]
    for i in range(len(all_probs[0])):
        ta=[]
        for j in range(5):
            try:
                val = all_probs[order.index(j) if order.index(j)!=len(order)-1 else order.index(j)-1][i][1 if order.index(j)!=len(order)-1 else 0]
                ta.append(val)
            except IndexError as e:
                print(f"IndexError at i={i}, j={j}, order.index(j)={order.index(j)}, len(order)={len(order)}, array shape: {all_probs[order.index(j) if order.index(j)!=len(order)-1 else order.index(j)-1].shape}")
                raise
        probs.append(ta)
    
    return probs


def rearranger(lst):
    """Reorder and save training output files based on a provided list.

    This helper walks through every fold's training outputs, computes a new
    order using ``order_fixer_pt2`` from the ``result_formatter`` module,
    applies ``reconstruct`` to each file to create normalized outputs and
    targets (prefixed with 'n'), and updates ``op_path`` in-place so callers
    can continue using the modified filenames. Primarily used for
    preprocessing before fusion analysis.

    The function may skip certain folds or models if:
    - The fold directory does not exist or does not end with "_fold{fold_no}".
    - The required files (outs, trgs, outs_prob) are missing for a model.
    - The order_fixer_pt2 or reconstruct functions fail (e.g., due to data inconsistencies).
    - For fold1 specifically, if model3's data has issues (e.g., mismatched lengths), the save may fail.

    Parameters
    ----------
    lst : list
        The list defining the desired output ordering for all folds.
    """
    op_path=getFusionOutputPath("")
    for i in op_path:
        print(f"Starting rearranger for fold {i}")
        # Only process fold1 for debugging
        # if i != "_fold1":
        #     print(f"Skipping fold {i} as per debug condition")
        #     continue            
        for idx,k in enumerate(op_path[i]["train_outputs"]):
            print(f"Processing model {idx+1}")
            try:
                # Compute the reordering based on the target labels and desired order
                trgs_data = np.load(k.replace("outs","trgs"))
                final_order = order_fixer_pt2(trgs_data, lst)
                
                # Reconstruct and save the normalized probability file
                prob_data = np.load(k.replace("outs","outs_prob"))
                reconstructed_prob = reconstruct(prob_data, final_order, lst)
                nouts_prob_path = k.replace("outs","nouts_prob")
                np.save(nouts_prob_path, reconstructed_prob)
                if not os.path.exists(nouts_prob_path):
                    print(f"ALARM: Failed to save {nouts_prob_path}")
                
                # Reconstruct and save the normalized output file
                outs_data = np.load(k)
                reconstructed_outs = reconstruct(outs_data, final_order, lst)
                nouts_path = k.replace("outs","nouts")
                np.save(nouts_path, reconstructed_outs)
                if not os.path.exists(nouts_path):
                    print(f"ALARM: Failed to save {nouts_path}")
                
                # Reconstruct and save the normalized target file
                reconstructed_trgs = reconstruct(trgs_data, final_order, lst)
                ntrgs_path = (k.replace("outs","trgs")).replace("trgs","ntrgs")
                np.save(ntrgs_path, reconstructed_trgs)
                if not os.path.exists(ntrgs_path):
                    print(f"ALARM: Failed to save {ntrgs_path}")
                
                # Update the paths to point to the normalized files
                op_path[i]["train_outputs"][idx]=k.replace("outs","nouts")
                op_path[i]["train_trgs"][0]=(k.replace("outs","trgs")).replace("trgs","ntrgs")
                print(f"Completed model {idx}")
                
            except Exception as e:
                print(f"Error processing model {idx} in fold {i}: {e}")
                continue


def get_save_dir():
    """Create and return a directory for storing combined results.

    Reads the "save_dir" field from ``config.json`` in the current working
    directory and makes a subdirectory called 'Combined' inside it. The
    newly created path is returned; an exception will be raised if the
    directory already exists.

    Returns
    -------
    str
        Absolute path of the created 'Combined' directory.
    """
    file_path="./config.json"
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    new_path=os.path.join(data["trainer"]["save_dir"],'Combined')
    os.mkdir(new_path)
    return new_path


def constructer(order):
    """Main entry point for fusing model outputs and saving reports.

    This function orchestrates the entire fusion workflow:

    1. Create the save directory using :func:`get_save_dir`.
    2. Retrieve paths for all folds via :func:`getFusionOutputPath`.
    3. For each fold, compute merged probability vectors and outputs using
       :func:`prob_merger` and :func:`merger`, store them in the combined
       directory, and accumulate global lists for evaluation.
    4. After processing all folds, compute classification reports and
       confusion matrices for both validation and training sets and save
       them as Excel files and Torch tensors. Filenames incorporate the
       experiment name from ``config.json``.

    Parameters
    ----------
    order : list of int
        Ordering permutation used for both probability and discrete output
        merging.
    """
    # set_expn_name(NAME,EPOCH)
    direc=get_save_dir()
    # print(op_path)

    op_path=getFusionOutputPath("n")
    all_outs = []
    all_trgs = []
    all_outst = []
    all_trgst = []
    for idx,i in enumerate(op_path):
        os.mkdir(os.path.join(direc,i))
        p=prob_merger(op_path[i],order,"train_prob")
        np.save(os.path.join(direc,i,"prob_train"),p)
        p=prob_merger(op_path[i],order,"val_prob")
        np.save(os.path.join(direc,i,"prob_val"),p)
        outs,trgs=merger(op_path[i],order,"val")
        np.save(os.path.join(direc,i,"outs_val"),outs)
        np.save(os.path.join(direc,i,"trgs_val"),trgs)
        all_outs.extend(outs)
        all_trgs.extend(trgs)
        outs,trgs=merger(op_path[i],order,"train")
        np.save(os.path.join(direc,i,"outs_train"),outs)
        np.save(os.path.join(direc,i,"trgs_train"),trgs)
        all_outst.extend(outs)
        all_trgst.extend(trgs)

    all_trgs = np.array(all_trgs).astype(int)
    all_outs = np.array(all_outs).astype(int)
    all_trgst = np.array(all_trgst).astype(int)
    all_outst = np.array(all_outst).astype(int)

    r = classification_report(all_trgs, all_outs, digits=6, output_dict=True)
    cm = confusion_matrix(all_trgs, all_outs)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(all_trgs, all_outs)
    df["accuracy"] = accuracy_score(all_trgs, all_outs)
    df = df * 100
    file_name = NAME + "_classification_report.xlsx"
    report_Save_path = os.path.join(direc, file_name)
    df.to_excel(report_Save_path)

    cm_file_name = NAME + "_confusion_matrix.torch"
    cm_Save_path = os.path.join(direc, cm_file_name)
    torch.save(cm, cm_Save_path)

    r = classification_report(all_trgst, all_outst, digits=6, output_dict=True)
    cm = confusion_matrix(all_trgst, all_outst)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(all_trgst, all_outst)
    df["accuracy"] = accuracy_score(all_trgst, all_outst)
    df = df * 100
    file_name = NAME + "_train_classification_report.xlsx"
    report_Save_path = os.path.join(direc, file_name)
    df.to_excel(report_Save_path)

    cm_file_name = NAME + "_train_confusion_matrix.torch"
    cm_Save_path = os.path.join(direc, cm_file_name)
    torch.save(cm, cm_Save_path)

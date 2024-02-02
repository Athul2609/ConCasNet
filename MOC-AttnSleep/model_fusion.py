import os
import json
import numpy as np
from sklearn.metrics import accuracy_score,cohen_kappa_score,classification_report,confusion_matrix
import pandas as pd
import torch
from result_formatter import reconstruct,order_fixer_pt2

#Experiment name
with open('.\config.json', "r") as json_file:
        data = json.load(json_file)
NAME=data["name"]
EPOCH =data["trainer"]["epochs"]
SAVED=data["trainer"]["save_dir"]
model_prefix=["model1","model2","model3","model4"]

def get_final_lists(folder_path,pref):
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
        print("none")
        import sys
        sys.exit()

def getFusionOutputPath(pref='n'):
    model_output_paths={}
    for fold_no in range(20):
        str_to_match=f"fold{fold_no}"
        str_to_match+=chr(92)
        str_to_match=str_to_match[:-1]
        model_output_paths[str_to_match]={"train_outputs":[],"val_outputs":[],"train_prob":[],"val_prob":[],"train_trgs":["hi"],"val_trgs":["hi"]}
        for name in model_prefix:
            folder_path=os.path.abspath(os.path.join(SAVED,name))
            for files in os.listdir(folder_path):
                if os.path.isdir(os.path.abspath(os.path.join(folder_path,files))) and str_to_match in files:
                    output_file,trgs_file,prob_file,train_output_file,train_trgs_file,train_prob_file=get_final_lists(os.path.abspath(os.path.join(folder_path,files)),pref)
                    model_output_paths[str_to_match]["val_outputs"].append(os.path.join(os.path.abspath(os.path.join(folder_path,files)),output_file))
                    model_output_paths[str_to_match]["train_outputs"].append(os.path.join(os.path.abspath(os.path.join(folder_path,files)),train_output_file))
                    model_output_paths[str_to_match]["train_prob"].append(os.path.join(os.path.abspath(os.path.join(folder_path,files)),train_prob_file))
                    model_output_paths[str_to_match]["val_prob"].append(os.path.join(os.path.abspath(os.path.join(folder_path,files)),prob_file))                    
                    model_output_paths[str_to_match]["train_trgs"][0]=(os.path.join(os.path.abspath(os.path.join(folder_path,files)),train_trgs_file))
                    model_output_paths[str_to_match]["val_trgs"][0]=(os.path.join(os.path.abspath(os.path.join(folder_path,files)),trgs_file))
                    break
    return model_output_paths

def merger(arr,order,mode):
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
        print(accuracy_score(outputs,trgs))
        print(outputs)
        print(outputs.shape)
        return outputs,trgs

def prob_merger(arr,order,mode):
    all_probs=[]
    for i in arr[mode]:
        all_probs.append(np.load(i))
    probs=[]
    for i in range(len(all_probs[0])):
        ta=[]
        for j in range(5):
            ta.append(all_probs[order.index(j) if order.index(j)!=len(order)-1 else order.index(j)-1][i][1 if order.index(j)!=len(order)-1 else 0])
        probs.append(ta)
    
    return probs


def rearranger(lst):
    op_path=getFusionOutputPath("")
    for i in op_path:
        for idx,k in enumerate(op_path[i]["train_outputs"]):
            final_order=order_fixer_pt2(np.load(k.replace("outs","trgs")),lst)
            np.save((k.replace("outs","nouts_prob")),reconstruct(np.load(k.replace("outs","outs_prob")),final_order,lst))

            np.save(k.replace("outs","nouts"),reconstruct(np.load(k),final_order,lst))
            np.save((k.replace("outs","trgs")).replace("trgs","ntrgs"),reconstruct(np.load(k.replace("outs","trgs")),final_order,lst))
            op_path[i]["train_outputs"][idx]=k.replace("outs","nouts")
            op_path[i]["train_trgs"][0]=(k.replace("outs","trgs")).replace("trgs","ntrgs")


def get_save_dir():
    file_path=".\config.json"
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    new_path=os.path.join(data["trainer"]["save_dir"],'Combined')
    os.mkdir(new_path)
    return new_path


def constructer(order):
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

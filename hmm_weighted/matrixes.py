import numpy as np

def breaker(labels_path,lent):
    labels=np.load(labels_path)
    labels_split=[]
    start=0
    for i in lent:
        labels_split.append(labels[start:start+i])
        start+=i
    return labels_split

def breaker_observed(labels_path,val_prob,weight):
    labels1=np.load(labels_path)
    val_prob=np.load(val_prob)
    labels=[]
    for i in range(len(labels1)):
        if(val_prob[i][int(labels1[i])]>weight[int(labels1[i])]):
            labels.append(2*labels1[i]+1)
        else:
            labels.append(2*labels1[i])
    labels=np.array(labels,dtype=labels1.dtype)
    # labels_split=[]
    # start=0
    # for i in lent:
    #     labels_split.append(labels[start:start+i])
    #     start+=i
    return labels

def dict_to_2d_matrix(input_dict):
    max_row = max(key[0] for key in input_dict.keys())
    max_col = max(key[1] for key in input_dict.keys())
    
    matrix = [[0] * (max_col + 1) for _ in range(max_row + 1)]

    for key, value in input_dict.items():
        matrix[key[0]][key[1]] = value
    
    return matrix

def normaliser(array):
    array = array.astype(np.float32)

    row_sums = array.sum(axis=1)


    for i in range(len(array)):
        if row_sums[i] != 0:
            array[i] /= row_sums[i]

    return array        

def transition_matrix(train_y):
    dic={}
    for i in range(5):
        for j in range(5):
            dic[(i,j)]=0
    for i in train_y:
        for j in range(len(i)-1):
            dic[(i[j],i[j+1])]+=1
    tm=dict_to_2d_matrix(dic)
    tm=np.array(tm)
    tm=normaliser(tm)
    return tm

def emission_matrix(outputs_path,labels_path,output_prob,weight):
    dic={}
    for i in range(5):
        for j in range(10):
            dic[(i,j)]=0
    labels=np.load(labels_path)
    outputs=np.load(outputs_path)
    output_prob=np.load(output_prob)
    # print(labels.shape)
    # print(outputs.shape)
    for i in range(len(labels)):
        if(output_prob[i][int(outputs[i])]>weight[int(outputs[i])]):
            dic[(labels[i],2*outputs[i]+1)]+=1
        else:
            dic[(labels[i],2*outputs[i])]+=1
    em=dict_to_2d_matrix(dic)
    em=np.array(em)
    em=normaliser(em)
    return em


# transition_matrix(r"C:\Users\athul\myfiles\Research\Sleep Stage\AttnSleep - Copy\saved\Exp1\03_09_2023_18_46_42_fold1\trgs_train_1.npy")
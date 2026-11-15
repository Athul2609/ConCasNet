import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import math

def order_fixer_pt1(arr,lst):
    """To find order of patients in the folds target outputs and return list of paths to patient input files in that order"""
    order=[]
    end=0
    while(end != len(arr)):
        l=[]
        for f in lst:
            try:
                i = np.load(f)["y"]
                # if len(i) > len(arr) - end:
                #     continue  # Skip if segment too short
                match = True
                for j in range(len(i)):
                    if i[j] != arr[end + j]:
                        match = False
                        break
                if match:
                    l.append(i)
                    order.append(f)
            except (KeyError, FileNotFoundError, ValueError) as e:
                print(f"Error loading {f}: {e}")
                continue
        if len(l) > 1:
            print("found more than one match")
            import sys
            sys.exit()
        elif len(l) == 1:
            end += len(l[0])
        else:
            print(f"No match found for segment starting at {end}")
            import sys
            sys.exit()  # Or handle differently
    return order

def order_fixer_pt2(arr,lst):
    """output a dictionary with paths to patient input files as keys and value as range of output array corresponding to that train file in the target outputs of the fold"""
    order=order_fixer_pt1(arr,lst)
    new_order=[0]
    for i in order:
        new_order.append(len(np.load(i)["y"])+new_order[-1])
    # print(new_order)
    final_order={}
    for idx,i in enumerate(order):
        final_order[i]=[new_order[idx],new_order[idx+1]]
    return final_order

def reconstruct(arr,final_order,lst):
    new_op=[]
    # print(len(final_order))
    for i in lst:
        try:
            for k in range(final_order[i][0],final_order[i][1]):
                new_op.append(arr[k])
        except KeyError:
            # print(f"Warning: {i} not found in final_order, skipping")
            pass
        except IndexError as e:
            print(f"IndexError for {i}: {e}")
            raise
    return np.array(new_op)
    


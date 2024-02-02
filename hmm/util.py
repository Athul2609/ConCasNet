import numpy as np
import torch
def calc_metrics(inp_directory):
        from sklearn.metrics import classification_report
        from sklearn.metrics import cohen_kappa_score
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        import pandas as pd
        import os
        from os import walk

        n_folds = 20
        all_outs = []
        all_trgs = []

        outs_list = []
        trgs_list = []
        for root, dirs, files in os.walk(inp_directory):
            for file in files:
                if "hmm_val" in file:
                     outs_list.append(os.path.join(root, file))
                if "trgs_val" in file:
                     trgs_list.append(os.path.join(root, file))

        if len(outs_list)==20:
            for i in range(len(outs_list)):
                outs = np.load(outs_list[i])
                trgs = np.load(trgs_list[i])
                all_outs.extend(outs)
                all_trgs.extend(trgs)

        all_trgs = np.array(all_trgs).astype(int)
        all_outs = np.array(all_outs).astype(int)

        r = classification_report(all_trgs, all_outs, digits=6, output_dict=True)
        cm = confusion_matrix(all_trgs, all_outs)
        df = pd.DataFrame(r)
        df["cohen"] = cohen_kappa_score(all_trgs, all_outs)
        df["accuracy"] = accuracy_score(all_trgs, all_outs)
        df = df * 100
        file_name = "hmm_classification_report.xlsx"
        report_Save_path = os.path.join(inp_directory, file_name)
        df.to_excel(report_Save_path)

        cm_file_name = "hmm_confusion_matrix.torch"
        cm_Save_path = os.path.join(inp_directory, cm_file_name)
        torch.save(cm, cm_Save_path)


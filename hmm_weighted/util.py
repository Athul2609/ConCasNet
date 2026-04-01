import numpy as np
import torch
def calc_metrics(inp_directory,weight,mode="val"):
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
        pattern=f"outs_{mode}_hmm_weighted"
        for root, dirs, files in os.walk(inp_directory):
            for file in files:
                if pattern in file:
                     outs_list.append(os.path.join(root, file))
                if f"trgs_{mode}" in file:
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
        file_name = f"hmm_classification_report_weighted_{mode}.xlsx"
        report_Save_path = os.path.join(inp_directory, file_name)
        df.to_excel(report_Save_path)
        print(f"saving results to {file_name}")
        cm_file_name = f"hmm_confusion_matrix_weighted{mode}.torch"
        cm_Save_path = os.path.join(inp_directory, cm_file_name)
        torch.save(cm, cm_Save_path)
        return accuracy_score(all_trgs, all_outs)




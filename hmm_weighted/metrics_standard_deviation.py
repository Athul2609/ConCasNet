import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix


def calc_metrics_std(inp_directory, mode="val", expected_folds=20, save=True):
    """Compute per-fold accuracy and Cohen's kappa, then return their mean and std.

    Searches for files named like `outs_{mode}_hmm_weighted*` and `trgs_{mode}*` under
    `inp_directory`. Loads matching numpy arrays per fold, computes per-fold
    accuracy and Cohen's kappa, and returns a summary plus lists of per-fold
    values and confusion matrices.

    Returns: (summary_dict, accuracies, cohens, confusion_matrices)
    """
    pattern_out = f"outs_{mode}_hmm_weighted"
    pattern_trg = f"trgs_{mode}"

    outs_list = []
    trgs_list = []
    for root, dirs, files in os.walk(inp_directory):
        for file in files:
            if pattern_out in file:
                outs_list.append(os.path.join(root, file))
            if pattern_trg in file:
                trgs_list.append(os.path.join(root, file))

    outs_list = sorted(outs_list)
    trgs_list = sorted(trgs_list)

    if len(outs_list) == 0 or len(trgs_list) == 0:
        raise FileNotFoundError("No matching outs/trgs files found in the input directory")

    pairs = min(len(outs_list), len(trgs_list))

    accuracies = []
    cohens = []
    cms = []
    per_fold = []

    for i in range(pairs):
        outs = np.load(outs_list[i])
        trgs = np.load(trgs_list[i])
        outs = np.array(outs).astype(int).reshape(-1)
        trgs = np.array(trgs).astype(int).reshape(-1)

        acc = accuracy_score(trgs, outs)
        coh = cohen_kappa_score(trgs, outs)
        cm = confusion_matrix(trgs, outs)

        accuracies.append(float(acc))
        cohens.append(float(coh))
        cms.append(cm)

        per_fold.append({"fold": i, "accuracy": float(acc), "cohen": float(coh), "n_samples": int(len(trgs))})

    accuracy_mean = float(np.mean(accuracies))
    accuracy_std = float(np.std(accuracies, ddof=0))
    cohen_mean = float(np.mean(cohens))
    cohen_std = float(np.std(cohens, ddof=0))

    summary = {
        "n_folds_found": pairs,
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "cohen_mean": cohen_mean,
        "cohen_std": cohen_std,
    }

    if save:
        file_name = f"hmm_metrics_summary_weighted_{mode}_standard_deviation.xlsx"
        report_path = os.path.join(inp_directory, file_name)
        df_per = pd.DataFrame(per_fold)
        df_summary = pd.DataFrame([summary])
        with pd.ExcelWriter(report_path) as writer:
            df_per.to_excel(writer, index=False, sheet_name="per_fold")
            df_summary.to_excel(writer, index=False, sheet_name="summary")

        # save confusion matrices as a torch tensor list
        cm_save_path = os.path.join(inp_directory, f"hmm_confusion_matrices_weighted_{mode}_standard_deviation.torch")
        torch.save(cms, cm_save_path)
        print(f"Saved summary to {file_name}")

    return summary, accuracies, cohens, cms


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, help="Input directory to scan for outs/trgs files")
    p.add_argument("-m", "--mode", default="val")
    args = p.parse_args()

    s, accs, coh, cms = calc_metrics_std(args.input, mode=args.mode, save=True)
    print(s)

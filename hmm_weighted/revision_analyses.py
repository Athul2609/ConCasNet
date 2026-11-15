"""
Analyses for the Revision 3 placeholders that need no training:

  R2  paired tests: selected cascading order vs the other five (EDF-20, SHHS)
  R6  paired tests: component-wise comparison (all three datasets; paper uses EDF-20)
  S4  threshold sensitivity (all thresholds 0.5, single shared threshold, per-fold selected)
  R3  where epochs exit the cascade + MACs / parameters (MOCAttnSleep vs AttnSleep)
  C3  fraction of recordings whose first retained epoch is Awake
  S1  model and training configuration
  S2  beta values and class weights at each cascade level

ConCasNet = per-fold train-selected thresholds (see train_selected_thresholds.py).
Statistics: two-sided Wilcoxon signed-rank test on the 20 fold-level values, Holm
correction within each family, and a 95% percentile bootstrap CI of the mean paired
difference (10,000 resamples of the 20 folds; for Sleep-EDF-20 each fold is one subject).

Run `python train_selected_thresholds.py --paper` first. This script expects the
paper's folder layout under <project> (the folder above this repository, or the
CONCASNET_PROJECT environment variable):
    <project>/Data/<DATA_DIRS below>/*.npz
    <project>/Results/...                              (Combined folders, see T.build_runs)
    <project>/Results/Train-Selected Thresholds/...   (written by train_selected_thresholds.py)
Outputs: <project>/Results/Revision Analyses/
Steps (s1 s2 c3 r3 r2 s4_and_r6 figure) can be run selectively:
    python revision_analyses.py s4_and_r6 figure
"""
import csv
import glob
import math
import os
import sys
from copy import deepcopy

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, cohen_kappa_score

import train_selected_thresholds as T

OUT = os.path.join(T.RESULTS, "Revision Analyses")
TST = T.OUT_ROOT
DATA = os.path.join(T.PROJECT, "Data")
MOC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MOC-AttnSleep"))
RNG = np.random.default_rng(123)
N_BOOT = 10_000

MAIN = {"Sleep-EDF-20": "Sleep-EDF-20 - main run",
        "Sleep-EDF-78": "Sleep-EDF-78 - main run",
        "SHHS": "SHHS - main run"}
DATA_DIRS = {"Sleep-EDF-20": "sleep-edf-13-fpz_cz", "Sleep-EDF-78": "sleep-edf-78", "SHHS": "SHHS"}
SELECTED = "3_0_2_4_1"
ALT_ORDERS = ["0_2_3_4_1", "0_3_2_4_1", "2_0_3_4_1", "2_3_0_4_1", "3_2_0_4_1"]
LABEL = dict(T.METHODS)


def short(code):
    return T.order_name(code).replace(" > ", "-")


def write_csv(name, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  wrote {name}")


def per_fold(run_folder, method_key):
    """{'accuracy': array(20), 'kappa': array(20)} for one method of one run."""
    rows = list(csv.DictReader(open(os.path.join(TST, run_folder, "per-fold test metrics.csv"))))
    rows = sorted([r for r in rows if r["method"] == LABEL[method_key]], key=lambda r: int(r["fold"]))
    assert len(rows) == 20, (run_folder, method_key, len(rows))
    return {m: np.array([float(r[m]) for r in rows]) for m in ("accuracy", "kappa")}


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def paired(a, b):
    d = a - b
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    p = 1.0 if np.allclose(d, 0) else wilcoxon(a, b).pvalue
    n_better = int((d > 0).sum())
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(p), n_better


def comparison_table(family, pairs):
    """pairs: list of (comparison label, dict_a, dict_b). Holm within each metric."""
    rows = []
    for metric, scale in (("accuracy", 1.0), ("kappa", 1.0)):
        res = [paired(a[metric] * scale, b[metric] * scale) for _, a, b in pairs]
        adj = holm([r[3] for r in res])
        for (label, _, _), (mean, lo, hi, p, nb), pa in zip(pairs, res, adj):
            unit = "pp" if metric == "accuracy" else ""
            rows.append(dict(family=family, comparison=label, metric=metric,
                             mean_difference=mean, ci95_low=lo, ci95_high=hi,
                             formatted=f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]{unit}" if metric == "accuracy"
                             else f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                             folds_where_first_is_better=f"{nb}/20",
                             p_wilcoxon=p, p_holm=float(pa), significant_at_005="yes" if pa < 0.05 else "no"))
    return rows


# --------------------------------------------------------------------------- R2
def r2():
    rows = []
    for ds, prefix in (("Sleep-EDF-20", "Sleep-EDF-20 order analysis - "), ("SHHS", "SHHS order analysis - ")):
        sel = per_fold(MAIN[ds], "concasnet_perfold")
        pairs = [(f"{short(SELECTED)} vs {short(c)}", sel, per_fold(prefix + short(c), "concasnet_perfold"))
                 for c in ALT_ORDERS]
        rows += comparison_table(f"R2 cascading order - {ds}", pairs)
    write_csv("R2 - cascading order paired tests.csv", rows)


# ------------------------------------------------------------------- S4 (+ R6)
def s4_and_r6():
    s4_rows, r6_rows = [], []
    for ds, run in MAIN.items():
        src = [r for r in T.build_runs() if r[0] == run][0][1]
        folds = T.load_folds(os.path.join(T.RESULTS, src))
        for f in folds:
            f["tm"] = T.transition_matrix(f["trgs_train"])
        y = [f["trgs_test"] for f in folds]

        def evaluate(th_for_fold):
            preds = [T.weighted_hmm(f, th_for_fold(f), "test") for f in folds]
            fa = np.array([100 * accuracy_score(a, b) for a, b in zip(y, preds)])
            fk = np.array([cohen_kappa_score(a, b) for a, b in zip(y, preds)])
            yy, pp = np.concatenate(y), np.concatenate(preds)
            return 100 * accuracy_score(yy, pp), cohen_kappa_score(yy, pp), {"accuracy": fa, "kappa": fk}

        acc05, k05, pf05 = evaluate(lambda f: [0.5] * 5)
        grid = [(t, *evaluate(lambda f, t=t: [t] * 5)[:2]) for t in T.GRID]
        summ = {r["method"]: r for r in csv.DictReader(open(os.path.join(TST, run, "summary - pooled test metrics.csv")))}
        s4_rows += [
            dict(dataset=ds, setting="Ordinary HMM (no confidence weighting)",
                 accuracy=float(summ[LABEL["ordinary_hmm"]]["accuracy"]), kappa=float(summ[LABEL["ordinary_hmm"]]["kappa"])),
            dict(dataset=ds, setting="All thresholds = 0.5 (identical to ordinary HMM, see README)", accuracy=acc05, kappa=k05),
            dict(dataset=ds, setting="Thresholds selected within each fold (training subjects)",
                 accuracy=float(summ[LABEL["concasnet_perfold"]]["accuracy"]), kappa=float(summ[LABEL["concasnet_perfold"]]["kappa"])),
            dict(dataset=ds, setting="Single shared threshold over 0.50-0.95: min",
                 accuracy=min(g[1] for g in grid), kappa=min(g[2] for g in grid)),
            dict(dataset=ds, setting="Single shared threshold over 0.50-0.95: max",
                 accuracy=max(g[1] for g in grid), kappa=max(g[2] for g in grid)),
        ] + [dict(dataset=ds, setting=f"  shared threshold = {t:.2f}", accuracy=a, kappa=k) for t, a, k in grid]

        cas, hmm_, ccn = (per_fold(run, m) for m in ("cascade_only", "ordinary_hmm", "concasnet_perfold"))
        r6_rows += comparison_table(f"R6 components - {ds}", [
            ("COVR-MOCAttnSleep+HMM vs COVR-MOCAttnSleep", hmm_, cas),
            ("ConCasNet vs COVR-MOCAttnSleep+HMM", ccn, hmm_),
        ])
        # thresholds fixed at 0.5 are not compared: s(e) >= 0.5 for every epoch, so every
        # prediction is "confident" and the model reduces to the ordinary HMM.
        print(f"  {ds}: thresholds-0.5 accuracy {acc05:.2f}")
    write_csv("S4 - threshold sensitivity.csv", s4_rows)
    write_csv("R6 - component-wise paired tests.csv", r6_rows)


# --------------------------------------------------------------------------- R3
def model_stats():
    import torch
    import torch.nn as nn
    from ptflops import get_model_complexity_info
    sys.path.insert(0, MOC)
    from model.model import (MRCNN, MRCNN_SHHS, MultiHeadedAttention, PositionwiseFeedForward,
                             TCE, EncoderLayer)

    class Net(nn.Module):
        def __init__(self, shhs, heads, n_classes):
            super().__init__()
            d_model, afr = (100 if shhs else 80), 30
            self.mrcnn = MRCNN_SHHS(afr) if shhs else MRCNN(afr)
            attn = MultiHeadedAttention(heads, d_model, afr)
            ff = PositionwiseFeedForward(d_model, 120, 0.1)
            self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr, 0.1), 2)
            self.fc = nn.Linear(d_model * afr, n_classes)

        def forward(self, x):
            z = self.tce(self.mrcnn(x))
            return self.fc(z.contiguous().view(z.shape[0], -1))

    stats = {}
    for name, shhs, heads, ncls, length in [("MOCAttnSleep (Sleep-EDF)", False, 2, 2, 3000),
                                            ("AttnSleep (Sleep-EDF)", False, 5, 5, 3000),
                                            ("MOCAttnSleep (SHHS)", True, 2, 2, 3750),
                                            ("AttnSleep (SHHS)", True, 5, 5, 3750)]:
        torch.manual_seed(0)
        m = Net(shhs, heads, ncls).eval()
        macs, params = get_model_complexity_info(m, (1, length), as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        stats[name] = (macs / 1e6, sum(p.numel() for p in m.parameters()) / 1e3)
    return stats


def r3():
    stats = model_stats()
    rows = [dict(item=f"{k}: MACs per epoch (M)", value=v[0]) for k, v in stats.items()] + \
           [dict(item=f"{k}: parameters (K)", value=v[1]) for k, v in stats.items()]
    order = [int(c) for c in SELECTED.split("_")]
    position = {order[0]: 1, order[1]: 2, order[2]: 3, order[3]: 4, order[4]: 4}
    for ds, run in MAIN.items():
        pred = np.load(os.path.join(TST, run, "test predictions - cascade_only.npy"))
        exits = np.array([position[int(p)] for p in pred])
        frac = [100 * float((exits == k).mean()) for k in (1, 2, 3, 4)]
        mean_cls = float(exits.mean())
        moc = stats["MOCAttnSleep (SHHS)" if ds == "SHHS" else "MOCAttnSleep (Sleep-EDF)"][0]
        rows += [dict(item=f"{ds}: % epochs exiting at classifier {k}", value=f) for k, f in zip((1, 2, 3, 4), frac)]
        rows += [dict(item=f"{ds}: mean classifiers evaluated per epoch", value=mean_cls),
                 dict(item=f"{ds}: mean MACs per epoch (M)", value=mean_cls * moc)]
    write_csv("R3 - cascade exits and computational cost.csv", rows)


# --------------------------------------------------------------------------- C3
def load_labels(ds):
    files = sorted(glob.glob(os.path.join(DATA, DATA_DIRS[ds], "*.npz")))
    return [np.load(f)["y"].astype(int) for f in files]


def c3():
    rows = []
    for ds in MAIN:
        ys = load_labels(ds)
        first = np.array([y[0] for y in ys])
        rows.append(dict(dataset=ds, recordings=len(ys),
                         starting_in_Awake=int((first == 0).sum()),
                         percent_starting_in_Awake=100 * float((first == 0).mean()),
                         first_epoch_distribution="; ".join(f"{T.STAGES[s]}={int((first == s).sum())}"
                                                            for s in range(5) if (first == s).any())))
    write_csv("C3 - recordings starting in Awake.csv", rows)


# --------------------------------------------------------------------------- S2
def s2():
    """Reproduces utils/util.py calc_class_weight. Counts come from train+test labels
    (all_ys in data_loaders.py), i.e. the whole dataset, so they are identical in every fold."""
    coeff = [1.5, 2, 1.5, 1, 1.5]
    order = [int(c) for c in SELECTED.split("_")]
    rows = []
    for ds in MAIN:
        y = np.concatenate(load_labels(ds))
        for level in range(4):
            target, rest = order[level], order[level + 1:]
            n1, n0 = int((y == target).sum()), int(np.isin(y, rest).sum())
            total = n0 + n1
            mu = [0.5 * (coeff[order[-1]] if level == 3 else 1), 0.5 * coeff[order[level]]]
            w = [round(max(1.0, math.log(m * total / n)) * m, 2) for m, n in zip(mu, (n0, n1))]
            rows.append(dict(dataset=ds,
                             level=f"{level + 1}: {T.STAGES[target]} vs {', '.join(T.STAGES[r] for r in rest)}",
                             n_target=n1, n_rest=n0, ratio_target_to_rest=f"1 : {n0 / n1:.2f}",
                             beta_target=mu[1], beta_rest=mu[0], w_target=w[1], w_rest=w[0]))
    write_csv("S2 - class weights per cascade level.csv", rows)


# --------------------------------------------------------------------------- S1
def s1():
    rows = [
        ("MRCNN branch 1 (small kernel)", "conv 50/6, 64 filters; maxpool 8/2; dropout 0.5; 2x conv 8/1, 128 filters; maxpool 4/4"),
        ("MRCNN branch 2 (wide kernel)", "conv 400/50, 64 filters; maxpool 4/2; dropout 0.5; 2x conv k/1, 128 filters (k = 7 Sleep-EDF, 6 SHHS); maxpool 2/2"),
        ("Activation", "GELU in MRCNN; ReLU in AFR and feed-forward"),
        ("AFR", "1 residual SE block, reduction ratio 16, 30 output channels"),
        ("TCE attention heads*", "2 (AttnSleep: 5)"),
        ("TCE model dim / feed-forward dim / layers", "80 Sleep-EDF, 100 SHHS / 120 / 2"),
        ("TCE causal conv kernel", "7"),
        ("Dropout", "0.5 (MRCNN), 0.1 (TCE)"),
        ("Output layer*", "2 classes (AttnSleep: 5)"),
        ("Optimizer", "Adam (AMSGrad), beta1 0.9, beta2 0.999, weight decay 1e-3"),
        ("Learning rate", "1e-3, reduced to 1e-4 after epoch 10"),
        ("Batch size", "128"),
        ("Training epochs", "50 per classifier, no early stopping"),
        ("Model selection", "predictions from the epoch with the highest held-out-fold accuracy (as in AttnSleep)"),
        ("Random seed", "123"),
        ("Parameters per classifier", "515.6 K (Sleep-EDF), 509.3 K (SHHS) - from saved checkpoints"),
    ]
    write_csv("S1 - model and training configuration.csv", [dict(setting=a, value=b) for a, b in rows])


# ----------------------------------------------------------------------- figure
def figure():
    """Replacement for Writing/resultsForPaper/resgen.py: Sleep-EDF-20, pooled values,
    error bars = SD of the 20 fold-level values (AttnSleep: published, no fold values)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = MAIN["Sleep-EDF-20"]
    summ = {r["method"]: r for r in csv.DictReader(open(os.path.join(TST, run, "summary - pooled test metrics.csv")))}
    names = ["AttnSleep", "COVR-MOCAttnSleep", "COVR-MOCAttnSleep+HMM", "ConCasNet"]
    keys = [None, "cascade_only", "ordinary_hmm", "concasnet_perfold"]
    colors = ["#B2DFDB", "#BBDEFB", "#FFCC80", "#FFAB91"]
    published = {"accuracy": 84.4, "macro_f1": 78.1, "kappa": 79.0}
    panels = [("accuracy", "Accuracy (%)"), ("macro_f1", "Macro F1 (%)"), ("kappa", "Cohen's kappa (x100)")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (metric, title) in zip(axs, panels):
        vals, errs = [], []
        for k in keys:
            if k is None:
                vals.append(published[metric]); errs.append(0); continue
            v = float(summ[LABEL[k]][metric]) * (100 if metric == "kappa" else 1)
            vals.append(v)
            if metric == "macro_f1":
                errs.append(0)
            else:
                pf = per_fold(run, k)["accuracy" if metric == "accuracy" else "kappa"]
                errs.append(float(np.std(pf)) * (100 if metric == "kappa" else 1))
        x = np.arange(len(names))
        ax.bar(x, vals, color=colors, edgecolor="black", yerr=errs, capsize=5)
        for xi, v, e in zip(x, vals, errs):
            ax.text(xi, v + e + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_title(title); ax.set_ylim(70, 96); ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "model_comparison_plot.png"), dpi=200, bbox_inches="tight")
    print("  wrote model_comparison_plot.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    steps = {f.__name__: f for f in (s1, s2, c3, r3, r2, s4_and_r6, figure)}
    for step in ([steps[a] for a in sys.argv[1:]] or steps.values()):
        print(f"== {step.__name__}", flush=True)
        step()


if __name__ == "__main__":
    main()

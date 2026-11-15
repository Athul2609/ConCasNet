"""
Confidence-weighted HMM of ConCasNet, with confidence thresholds selected on the
TRAINING subjects of each fold only (the procedure reported in the paper).

Input: the `Combined` folder written by MOC-AttnSleep/kfoldcv.py, which holds fold0..fold19,
each with outs/trgs/prob for the fold's training subjects (outs_train, trgs_train,
prob_train) and held-out subjects (outs_val, trgs_val, prob_val).

For every fold:
  * transition matrix: counted from the concatenated training hypnograms
  * emission matrix:   cascade predictions on the training subjects vs their true stages;
                       each prediction is split into confident / not confident
                       (probability of the predicted stage strictly > that stage's threshold)
  * start state:       Awake
  * thresholds:        coordinate search (Awake, N1, N2, N3, REM in turn; grid 0.50-0.95,
                       then 0.90-0.99), scored by HMM accuracy on the fold's training subjects
  * the held-out recordings are decoded once, as one sequence, with the selected thresholds

Four methods are reported: cascade only, ordinary HMM (no confidence split),
ConCasNet with one global threshold set (chosen on the pooled training decodes of all
folds), and ConCasNet with per-fold thresholds (the paper's ConCasNet).

Usage:
    python train_selected_thresholds.py <Combined folder> <output folder> [order, e.g. 3_0_2_4_1]
    python train_selected_thresholds.py --paper [run name ...]
        re-runs the paper's runs from <project>/Results (see build_runs); <project> is the
        folder above this repository, or the CONCASNET_PROJECT environment variable.
"""
import csv
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
from hmmlearn import hmm
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

# Folder that holds Results/ and Data/ (only used by --paper and revision_analyses.py)
PROJECT = os.environ.get("CONCASNET_PROJECT",
                         os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
RESULTS = os.path.join(PROJECT, "Results")
OUT_ROOT = os.path.join(RESULTS, "Train-Selected Thresholds")

STAGES = ["Awake", "N1", "N2", "N3", "REM"]          # label index 0..4
GRID = [k / 100 for k in range(50, 100, 5)]          # 0.50 .. 0.95
FINE_OFFSETS = [l / 100 for l in range(-5, 5)]       # -0.05 .. +0.04


def order_name(code):
    return " > ".join(STAGES[int(c)] for c in code.split("_"))


def build_runs():
    """The paper's runs: (run name, Combined folder relative to RESULTS, cascade order)."""
    runs = [
        ("Sleep-EDF-20 - main run", "ConCasNet Results/3_0_2_4_1 - sleep-edf-20/Combined", "3_0_2_4_1"),
        ("Sleep-EDF-78 - main run", "ConCasNet Results/3_0_2_4_1 - sleep-edf-78/Combined", "3_0_2_4_1"),
        ("SHHS - main run", "ConCasNet Results/3_0_2_4_1 - SHHS/Combined", "3_0_2_4_1"),
        ("Sleep-EDF-20 Pz-Oz - electrode sensitivity",
         "Results for pz-oz sleep edf 20 sensitivity analysis/Combined", "3_0_2_4_1"),
    ]
    for code in ["0_2_3_4_1", "0_3_2_4_1", "2_0_3_4_1", "2_3_0_4_1", "3_2_0_4_1"]:
        runs.append((f"Sleep-EDF-20 order analysis - {order_name(code).replace(' > ', '-')}",
                     f"SLEEP-EDF-20 Order Analysis/{code}/Combined", code))
    for code in ["0_2_3_4_1", "0_3_2_4_1", "2_0_3_4_1", "2_3_0_4_1", "3_2_0_4_1"]:
        runs.append((f"SHHS order analysis - {order_name(code).replace(' > ', '-')}",
                     f"SHHS Order Analysis/{code}/Combined", code))
    return runs


# ----------------------------------------------------------------------------- data
def load_folds(combined_dir):
    def one(fold_dir, prefix):
        hits = [f for f in os.listdir(fold_dir)
                if f.startswith(prefix) and f.endswith(".npy") and "hmm" not in f]
        if len(hits) != 1:
            raise RuntimeError(f"{fold_dir}: expected one {prefix}*.npy, found {hits}")
        return np.load(os.path.join(fold_dir, hits[0]))

    folds = []
    for d in os.listdir(combined_dir):
        m = re.search(r"fold(\d+)$", d)
        if m and os.path.isdir(os.path.join(combined_dir, d)):
            fd = os.path.join(combined_dir, d)
            folds.append(dict(
                idx=int(m.group(1)),
                outs_train=one(fd, "outs_train").astype(int),
                trgs_train=one(fd, "trgs_train").astype(int),
                prob_train=one(fd, "prob_train"),
                outs_test=one(fd, "outs_val").astype(int),
                trgs_test=one(fd, "trgs_val").astype(int),
                prob_test=one(fd, "prob_val"),
            ))
    folds.sort(key=lambda f: f["idx"])
    if len(folds) != 20:
        raise RuntimeError(f"{combined_dir}: expected 20 folds, found {len(folds)}")
    for f in folds:
        # confidence of the predicted stage (same quantity matrixes.py thresholds)
        f["conf_train"] = f["prob_train"][np.arange(len(f["outs_train"])), f["outs_train"]]
        f["conf_test"] = f["prob_test"][np.arange(len(f["outs_test"])), f["outs_test"]]
    return folds


# ------------------------------------------------------------------- HMM (as before)
def row_normalise(counts):
    counts = counts.astype(np.float64)
    s = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, s, out=np.zeros_like(counts), where=s != 0)


def transition_matrix(trgs):
    # matrixes.transition_matrix: consecutive pairs over the concatenated training labels
    counts = np.zeros((5, 5))
    np.add.at(counts, (trgs[:-1], trgs[1:]), 1)
    return row_normalise(counts)


def observations(outs, conf, thresholds):
    # matrixes.breaker_observed: 2*stage + 1 if confident (strictly above threshold)
    th = np.asarray(thresholds)[outs]
    return 2 * outs + (conf > th).astype(int)


def emission_matrix(trgs, obs, n_symbols):
    counts = np.zeros((5, n_symbols))
    np.add.at(counts, (trgs, obs), 1)
    return row_normalise(counts)


def decode(tm, em, obs):
    model = hmm.CategoricalHMM(n_components=5)
    model.startprob_ = np.array([1.0, 0.0, 0.0, 0.0, 0.0])   # as in hmm_input.py
    model.transmat_ = tm
    model.emissionprob_ = em
    model.n_features = em.shape[1]
    return model.predict(obs.reshape(-1, 1))


def weighted_hmm(fold, thresholds, on):
    """Fit on the fold's training data, decode `on` ('train' or 'test')."""
    obs_train = observations(fold["outs_train"], fold["conf_train"], thresholds)
    em = emission_matrix(fold["trgs_train"], obs_train, 10)
    if on == "train":
        return decode(fold["tm"], em, obs_train)
    obs_test = observations(fold["outs_test"], fold["conf_test"], thresholds)
    return decode(fold["tm"], em, obs_test)


def ordinary_hmm_test(fold):
    em = emission_matrix(fold["trgs_train"], fold["outs_train"], 5)
    return decode(fold["tm"], em, fold["outs_test"])


# ------------------------------------------------------------- threshold search
def coordinate_search(score):
    """Same search as hmm_input.py: start at 0.5, one stage at a time in label order,
    coarse grid 0.50-0.95, then the fine pass of hmm_input.py, which (because best_n
    is taken from the last grid value) scans 0.90-0.99. Strict improvement only."""
    best = [0.5] * 5
    best_score = -1.0
    for i in range(5):
        for k in GRID:
            cand = best.copy(); cand[i] = k
            s = score(cand)
            if s > best_score:
                best_score, best = s, cand
        last = GRID[-1]
        for off in FINE_OFFSETS:
            cand = best.copy(); cand[i] = round(last + off, 2)
            s = score(cand)
            if s > best_score:
                best_score, best = s, cand
    return best, best_score


def search_global(folds):
    cache = {}

    def score(th):
        key = tuple(th)
        if key not in cache:
            correct = total = 0
            for f in folds:
                pred = weighted_hmm(f, th, "train")
                correct += int((pred == f["trgs_train"]).sum())
                total += len(pred)
            cache[key] = correct / total
        return cache[key]
    return coordinate_search(score)


def search_one_fold(fold):
    cache = {}

    def score(th):
        key = tuple(th)
        if key not in cache:
            pred = weighted_hmm(fold, th, "train")
            cache[key] = float((pred == fold["trgs_train"]).mean())
        return cache[key]
    return coordinate_search(score)


# --------------------------------------------------------------------- metrics
def metrics(y, p):
    _, _, f1, _ = precision_recall_fscore_support(y, p, labels=range(5), zero_division=0)
    return dict(accuracy=100 * accuracy_score(y, p),
                macro_f1=100 * f1_score(y, p, labels=range(5), average="macro", zero_division=0),
                kappa=cohen_kappa_score(y, p),
                **{f"F1_{s}": 100 * v for s, v in zip(STAGES, f1)})


def write_csv(path, rows, header):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


# ------------------------------------------------------------------------ main
METHODS = [
    ("cascade_only", "COVR-MOCAttnSleep (cascade only, no HMM)"),
    ("ordinary_hmm", "COVR-MOCAttnSleep + ordinary HMM"),
    ("concasnet_global", "ConCasNet - thresholds selected on training subjects (one set for all folds)"),
    ("concasnet_perfold", "ConCasNet - thresholds selected on training subjects (separate set per fold)"),
]


def run(name, src, out, order_code):
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    print(f"\n=== {name}\n    source: {src}", flush=True)

    folds = load_folds(src)
    for f in folds:
        f["tm"] = transition_matrix(f["trgs_train"])

    th_global, train_acc_global = search_global(folds)
    print(f"    global thresholds {th_global} (train acc {100*train_acc_global:.2f}%)", flush=True)

    preds = {m: [] for m, _ in METHODS}
    th_rows = [dict(fold="all folds (global)", **dict(zip(STAGES, th_global)),
                    train_accuracy=100 * train_acc_global)]
    for f in folds:
        th_f, tr_acc = search_one_fold(f)
        th_rows.append(dict(fold=f["idx"], **dict(zip(STAGES, th_f)), train_accuracy=100 * tr_acc))
        preds["cascade_only"].append(f["outs_test"])
        preds["ordinary_hmm"].append(ordinary_hmm_test(f))
        preds["concasnet_global"].append(weighted_hmm(f, th_global, "test"))
        preds["concasnet_perfold"].append(weighted_hmm(f, th_f, "test"))
    trgs = [f["trgs_test"] for f in folds]
    y_all = np.concatenate(trgs)

    summary, per_fold = [], []
    for m, label in METHODS:
        p_all = np.concatenate(preds[m])
        fold_acc = [100 * accuracy_score(y, p) for y, p in zip(trgs, preds[m])]
        fold_k = [cohen_kappa_score(y, p) for y, p in zip(trgs, preds[m])]
        summary.append(dict(method=label, **metrics(y_all, p_all),
                            fold_accuracy_mean=float(np.mean(fold_acc)),
                            fold_accuracy_sd=float(np.std(fold_acc)),
                            fold_kappa_mean=float(np.mean(fold_k)),
                            fold_kappa_sd=float(np.std(fold_k))))
        for f, y, p in zip(folds, trgs, preds[m]):
            per_fold.append(dict(method=label, fold=f["idx"], n_test_epochs=len(y),
                                 accuracy=100 * accuracy_score(y, p),
                                 kappa=cohen_kappa_score(y, p)))
        cm = confusion_matrix(y_all, p_all, labels=range(5))
        with open(os.path.join(out, f"confusion matrix - {m}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["true \\ predicted"] + STAGES)
            for s, row in zip(STAGES, cm):
                w.writerow([s] + list(row))
        np.save(os.path.join(out, f"test predictions - {m}.npy"), p_all)
    np.save(os.path.join(out, "test labels.npy"), y_all)

    hdr = list(summary[0].keys())
    write_csv(os.path.join(out, "summary - pooled test metrics.csv"), summary, hdr)
    write_csv(os.path.join(out, "per-fold test metrics.csv"), per_fold, list(per_fold[0].keys()))
    write_csv(os.path.join(out, "selected thresholds.csv"), th_rows, list(th_rows[0].keys()))
    with open(os.path.join(out, "run info.txt"), "w") as fh:
        fh.write(f"Run: {name}\nSource predictions: {src}\nCascading order: {order_name(order_code)}\n"
                 f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n"
                 f"Script: ConCasNet/hmm_weighted/train_selected_thresholds.py\n"
                 f"Test epochs: {len(y_all)}\n")
    for r in summary:
        print(f"    {r['method'][:70]:70s} ACC {r['accuracy']:.2f}  MF1 {r['macro_f1']:.2f}  k {r['kappa']:.4f}",
              flush=True)
    print(f"    done in {time.time()-t0:.0f}s", flush=True)
    return [dict(run=name, cascading_order=order_name(order_code), **r) for r in summary]


def main():
    args = sys.argv[1:]
    if args and args[0] != "--paper":
        if len(args) not in (2, 3):
            sys.exit(__doc__)
        src, out = args[0], args[1]
        run(os.path.basename(os.path.normpath(out)), src, out, args[2] if len(args) == 3 else "3_0_2_4_1")
        return
    if not args:
        sys.exit(__doc__)

    runs = build_runs()
    if len(args) > 1:
        runs = [r for r in runs if r[0] in args[1:]]
    os.makedirs(OUT_ROOT, exist_ok=True)
    all_rows = []
    for name, rel, code in runs:
        all_rows += run(name, os.path.join(RESULTS, rel), os.path.join(OUT_ROOT, name), code)
        overview = os.path.join(OUT_ROOT, "ALL RUNS - summary.csv")
        existing = []
        if os.path.exists(overview):
            with open(overview, newline="") as fh:
                existing = [r for r in csv.DictReader(fh) if r["run"] not in {x["run"] for x in all_rows}]
        rows = existing + all_rows
        write_csv(overview, rows, list(all_rows[0].keys()))


if __name__ == "__main__":
    main()

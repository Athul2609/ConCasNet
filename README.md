# ConCasNet

Code for **"Single-Channel Electroencephalogram Sleep Staging via Cascading One-vs-Rest
Networks and Confidence-Weighted Hidden Markov Modeling"**.

ConCasNet stages 30-s epochs of single-channel EEG in two parts:

1. **Cascade (COVR-MOCAttnSleep).** Four binary classifiers are applied in sequence. Each
   one separates a single stage from the stages that remain. With the order used in the
   paper, N3 → Awake → N2 → REM → N1, classifier 1 asks "N3 or not?", classifier 2 asks
   "Awake or not?", and so on. The last classifier separates REM from N1. Every classifier
   is an AttnSleep network ([Eldele et al., 2021](https://github.com/emadeldeen24/AttnSleep))
   reduced to two outputs (MOCAttnSleep).
2. **Confidence-weighted HMM.** Each predicted stage is labelled confident or not confident
   by comparing its probability with a per-stage threshold. The resulting sequence is decoded
   with a hidden Markov model. Transitions, emissions and thresholds are estimated on the
   training subjects of each fold only.

Training uses subject-wise 20-fold cross-validation, with the same folds as AttnSleep.

## Repository layout

| Path | Contents |
|---|---|
| `MOC-AttnSleep/` | The cascade: data preparation, model, training, fusion of the four classifiers |
| `MOC-AttnSleep/kfoldcv.py` | Entry point: trains the 4 classifiers × 20 folds, then fuses them |
| `MOC-AttnSleep/model/model.py` | MOCAttnSleep. **Contains the Sleep-EDF / SHHS switch.** |
| `MOC-AttnSleep/utils/util.py` | Fold assignment (AttnSleep's fixed permutations in `utils/r_permute_*.npy`) and class weights |
| `MOC-AttnSleep/compute_model_stats.py` | Parameters and MACs of one classifier |
| `hmm_weighted/train_selected_thresholds.py` | Confidence-weighted HMM, with thresholds selected on training subjects. It also reports the cascade alone and an ordinary HMM. |
| `hmm_weighted/revision_analyses.py` | Statistical tests and supplementary analyses of the paper |
| `supplementary_details/` | Folds used, thresholds and HMM parameters per fold (see below) |

## Setup

```bash
conda create -n concasnet python=3.9
conda activate concasnet
pip install -r requirements.txt
```

The code was run on Windows with one NVIDIA GPU. It uses only portable paths, so it should
also run on Linux.

## 1. Prepare the data

**Sleep-EDF.** Download the Sleep Cassette recordings from
[PhysioNet](https://physionet.org/content/sleep-edfx/). For Sleep-EDF-20 you can use
`prepare_datasets/download_edf20.sh` or `download_edf20.ps1`. Sleep-EDF-78 is the full set of
Sleep Cassette recordings.

**SHHS.** Request access at the [NSRR](https://sleepdata.org/datasets/shhs). The paper uses
the 329 SHHS-1 subjects in `prepare_datasets/selected_shhs1_files.txt`, the same subjects as
AttnSleep.

```bash
cd MOC-AttnSleep/prepare_datasets
# Sleep-EDF-20 / 78 (100 Hz, Fpz-Cz)
python prepare_physionet.py --data_dir <edf folder> --output_dir <data>/sleep-edf-20 --select_ch "EEG Fpz-Cz"
# SHHS (125 Hz, C4-A1)
python prepare_shhs.py --data_dir <shhs1 edfs> --ann_dir <shhs1 profusion annotations> --output_dir <data>/SHHS --select_ch "EEG C4-A1"
```

Each recording becomes one `.npz` file holding `x` (epochs × samples) and `y` (stages:
0 Awake, 1 N1, 2 N2, 3 N3, 4 REM).

> **The folder name selects the dataset.** `kfoldcv.py` uses the SHHS folds when the data
> path contains `SHHS` (capital letters). It uses the Sleep-EDF-78 folds when the path
> contains `78`, and the Sleep-EDF-20 folds otherwise. A folder named `shhs` would therefore
> be split as Sleep-EDF-20.

## 2. Train the cascade and fuse the classifiers

Run this from inside `MOC-AttnSleep/`, because `config.json` and `utils/r_permute_*.npy` are
read with relative paths:

```bash
cd MOC-AttnSleep
python kfoldcv.py --order "3 0 2 4 1" --np_data_dir <data>/sleep-edf-20 --save_dir saved --device 0
```

- `--order` gives the cascade order as stage indices. `"3 0 2 4 1"` is N3 → Awake → N2 → REM
  → N1, the order used in the paper. The paper also compares `"0 2 3 4 1"`, `"0 3 2 4 1"`,
  `"2 0 3 4 1"`, `"2 3 0 4 1"` and `"3 2 0 4 1"`.
- Training settings are in `config.json`: Adam with AMSGrad, learning rate 1e-3 reduced to
  1e-4 after epoch 10, weight decay 1e-3, batch size 128, and 50 epochs per classifier.
  `kfoldcv.py` sets `name` and `save_dir` in `config.json` itself.
- For each fold, the predictions kept are those of the epoch with the highest held-out
  accuracy, as in AttnSleep.

Output, under `saved/3_0_2_4_1/`:

| Folder | Contents |
|---|---|
| `model1/` … `model4/` | One folder per classifier and fold, with the binary predictions and probabilities |
| `Combined/fold0` … `fold19/` | Fused 5-class predictions (`outs_*`), labels (`trgs_*`) and per-stage probabilities (`prob_*`) for the training (`*_train`) and held-out (`*_val`) subjects |
| `Combined/*_classification_report.xlsx` | Pooled results of the cascade without the HMM |

The `Combined` folder must not already exist; delete it before re-running the fusion.

## 3. Confidence-weighted HMM

```bash
cd hmm_weighted
python train_selected_thresholds.py ../MOC-AttnSleep/saved/3_0_2_4_1/Combined results/sleep-edf-20
```

The script writes:
- `summary - pooled test metrics.csv`: accuracy, macro-F1, κ and per-stage F1 for the cascade
  alone, the ordinary HMM, and ConCasNet. ConCasNet is reported with one threshold set for
  all folds and with **per-fold thresholds**, which is the paper's ConCasNet.
- `per-fold test metrics.csv`
- `selected thresholds.csv`
- the confusion matrices

The test folds are decoded only once, after the thresholds are chosen.

`revision_analyses.py` reproduces the paper's paired Wilcoxon tests, bootstrap confidence
intervals, threshold sensitivity, computational cost and class-weight tables. It expects the
authors' folder layout, which is described at the top of the file.

## Running on SHHS instead of Sleep-EDF

The pipeline is the same. Only the items marked **change** differ:

| | Sleep-EDF-20 / Sleep-EDF-78 | SHHS |
|---|---|---|
| Channel | Fpz-Cz (Pz-Oz also tested on Sleep-EDF-20) | C4-A1 |
| Sampling rate, samples per epoch | 100 Hz, 3000 | 125 Hz, 3750 |
| Preparation script | `prepare_physionet.py` | `prepare_shhs.py`, reading the 329 subjects in `selected_shhs1_files.txt` |
| Data folder name | anything; include `78` for Sleep-EDF-78 | **change:** must contain `SHHS` |
| Folds | `load_folds_data` with `r_permute_20.npy` / `r_permute_78.npy`; both nights of a subject in the same fold | `load_folds_data_shhs` with `r_permute_shhs.npy`; one recording per subject |
| `model/model.py`, `d_model` | `80` | **change:** `100` |
| `model/model.py`, feature extractor | `MRCNN` | **change:** `MRCNN_SHHS` (kernel 6 instead of 7 in the wide branch) |
| Parameters per classifier | 515.6 K | 509.3 K |
| `compute_model_stats.py -i` | `1,3000` | **change:** `1,3750` |
| Class weights, cascade order, training settings, HMM | same | same |

**Steps for SHHS:**

1. In `MOC-AttnSleep/model/model.py`, edit the two lines marked `DATASET SWITCH`:
   ```python
   d_model = 100                              # was 80
   self.mrcnn = MRCNN_SHHS(afr_reduced_cnn_size)  # was MRCNN(...)
   ```
2. Run `kfoldcv.py` with a data folder whose path contains `SHHS`:
   ```bash
   python kfoldcv.py --order "3 0 2 4 1" --np_data_dir <data>/SHHS --save_dir saved_SHHS
   ```
3. Run the HMM step on `saved_SHHS/3_0_2_4_1/Combined` as above. Nothing in the HMM changes.

Change the two lines back before training on Sleep-EDF again. A model built for one dataset
cannot load the other's checkpoints.

## Supplementary details

`supplementary_details/` holds the files referred to in the paper's data-availability
statement:

| File | Contents |
|---|---|
| `folds used.csv` | The held-out recordings of each of the 20 folds, for Sleep-EDF-20, Sleep-EDF-78 and SHHS. The training set of a fold is every other recording. The SHHS rows list the 329 subjects used. Pz-Oz uses the same folds as Sleep-EDF-20. |
| `confidence thresholds per fold.csv` | Awake, N1, N2, N3 and REM thresholds selected in each fold |
| `hmm parameters per fold.csv` | Transition matrix and emission matrices (confidence-weighted and ordinary HMM) of each fold. One row per entry, with columns from (true stage), to, probability. |

The folds are those produced by `utils/util.py`. For every fold, the listed held-out
recordings reproduce the saved test labels exactly.

## Acknowledgement

MOC-AttnSleep is built on [AttnSleep](https://github.com/emadeldeen24/AttnSleep) (MIT
License). The architecture, data preparation, fold permutations and training framework come
from that repository.

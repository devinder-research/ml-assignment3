# Machine Learning Assignment 3 — CIFAR10

**Devinder · P26AI0004 · Split seed 260004**

The repository contains the implementations for Q1–Q3 and the Q4 bonus,
the fixed CIFAR10 split, experiment logs and configurations, and the source
and figures for the two-column report. The submission PDF is built after
the final source commit so page 1 can identify that exact commit.

## Setup

From the repository root, with Python 3.10:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The original runs used Apple M2/MPS. The scripts choose CUDA, MPS or CPU
according to availability. The recorded results are for one split and run
seed. Repeated runs on different backends can differ numerically and in elapsed
time. The full run takes several hours on CPU and downloads official CIFAR10
on first use. Data, feature caches, and model checkpoints are generated locally.

## Reproduction

Build the report from the committed results and figures in a clean checkout:

```bash
bash reproduce.sh report
```

Recompute all results and rebuild the report:

```bash
bash reproduce.sh full
```

Both commands write `report/main.pdf` and use the current Git commit hash.
The destination GitHub URL for this submission is
`https://github.com/devinder-research/ml-assignment3`. To use a different
repository URL, regenerate the standalone PDF with
`bash reproduce.sh report --repo-url https://github.com/OWNER/REPO`.

The full run uses these commands in order. Each works from the repository root:

| Stage | Scripts |
|---|---|
| Dataset and split | `python scripts/make_split.py` |
| Q1 classification and memorization | `python scripts/q1a_comparison.py`, `python scripts/q1b_memorization.py` |
| Q2 CNN, permutation and shifts | `python scripts/q2a_cnn_vs_mlp.py`, `python scripts/q2b_permuted.py`, `python scripts/q2c_translation.py` |
| Q3 autoencoders and representations | `python scripts/q3a_autoencoders.py`, `python scripts/q3b_denoising.py`, `python scripts/q3a_recon_grid.py`, `python scripts/q3c_representations.py` |
| Q4 bonus | `python scripts/q4_features.py`, the grid below, then `python scripts/q4b_hard_pairs.py` and `python scripts/q4c_clustering.py` |
| Figures and report | `python scripts/make_figures.py`, then `python report/build_report.py --commit "$(git rev-parse HEAD)"` |

Q4(a) from scratch:

```bash
for kernel in linear poly rbf; do
  for c in 0.01 0.1 1 10 100; do
    python scripts/q4a_svm_sweep.py --kernel "$kernel" --C "$c"
  done
done
python scripts/q4a_svm_sweep.py --finalize
```

Each grid fit writes a separate result. Finalization requires all 15 fits
and selects the kernel and C by validation accuracy, then by smaller C
and fewer support vectors. The selected SVM and its test predictions are
reused by Q4(b). To redo only finalization with the archived sweep, keep
the original feature cache and pass `--finalize-only-recorded`; the command
checks that the selected row's validation accuracy and support count match.

## Experiment files

- `src/` defines data loading, the shared models, training and metrics.
- `scripts/` contains the experiment configurations and entry points.
- `split/` contains the stratified 45,000/5,000 split (4,500 train and
  500 validation images per class) and the fixed Q2 pixel permutation.
- `results/` contains epoch logs, final metrics, the full Q4 sweep and
  the fixed 9,000-image stratified SVM subset.
- `figs/` contains report plots and image grids. `report/build_report.py`
  assembles the report directly from the saved results.

The Q2(a) models are reused in Q2(c), and its CNN is reused as the fixed
classifier in Q3(b). Q3(a)'s encoders are frozen for Q3(c); its best
validation-MSE autoencoder supplies the Q3(b) baseline and the fixed Q4
representation. Q4(c) compares its assignments with the selected Q4(a) SVM.



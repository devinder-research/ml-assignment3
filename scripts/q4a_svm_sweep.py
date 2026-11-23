"""Q4(a): SVM over the fixed representation - C x kernel sweep.

Fixed choices (before any result is seen):
  * representation: frozen bottleneck features of the validation-selected
    clean autoencoder (see q4_features.py) - unchanged across this question;
  * training set: the fixed stratified 9,000-sample subset of the 45,000-sample
    train split (seed 260004), chosen BEFORE any model selection, used for
    every SVM fit;
  * validation set: the fixed 5,000-image official validation split;
  * preprocessing: StandardScaler fit on the 9,000 subset ONLY;
  * kernels: linear, RBF (gamma='scale'), poly (degree 3, gamma='scale');
  * C values (5 values spanning 4 orders of magnitude): 0.01, 0.1, 1, 10, 100;
  * solver: sklearn SVC (libsvm), probability=False.

Run each grid cell with --kernel and --C, then use --finalize to select
by validation accuracy and evaluate the selected model on the official test
set. --finalize-only-recorded uses the shipped sweep and the original feature
cache to reproduce post-training analyses.
"""
import sys
import csv
import json
import os
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils import (DATA_DIR, RESULTS_DIR, save_json, SEED, ensure_dir)

C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
KERNELS = ["linear", "rbf", "poly"]
POLY_DEGREE = 3
Q4DIR = RESULTS_DIR / "q4"
SWEEP_FILE = Q4DIR / "sweep.csv"
RUN_DIR = DATA_DIR / "q4_sweep"
MODEL_FILE = DATA_DIR / "q4_selected.joblib"
TEST_FILE = DATA_DIR / "q4_selected_test.npz"


def make_svm(kernel, C):
    return SVC(C=C, kernel=kernel, degree=POLY_DEGREE, cache_size=1000,
               decision_function_shape="ovo")


def cell_file(kernel, C):
    return RUN_DIR / f"{kernel}-{C:g}.json"


def load_xy(include_test=False):
    ftr = np.load(DATA_DIR / "q4_features_train.npy", mmap_mode="r")
    fva = np.load(DATA_DIR / "q4_features_val.npy")
    ytr = np.load(DATA_DIR / "q4_labels_train.npy")
    yva = np.load(DATA_DIR / "q4_labels_val.npy")
    sub = np.load(RESULTS_DIR / "q4" / "svm_subset_seed260004.npz")["subset_idx"]
    train = ftr[sub]
    scaler = StandardScaler().fit(train)
    result = (scaler.transform(train), ytr[sub], scaler.transform(fva), yva)
    if not include_test:
        return result
    fte = np.load(DATA_DIR / "q4_features_test.npy", mmap_mode="r")
    yte = np.load(DATA_DIR / "q4_labels_test.npy")
    return (*result, scaler.transform(fte), yte)


def fit_row(kernel, C, Xs, ys, Xv, yva):
    t0 = time.time()
    clf = make_svm(kernel, C)
    clf.fit(Xs, ys)
    dt = time.time() - t0
    n_sv = int(clf.n_support_.sum())
    return {
        "kernel": kernel, "C": C,
        "train_acc": accuracy_score(ys, clf.predict(Xs)),
        "val_acc": accuracy_score(yva, clf.predict(Xv)),
        "n_support_vectors": n_sv,
        "sv_fraction": n_sv / len(Xs),
        "fit_seconds": round(dt, 1),
    }


def task(kernel, C):
    Xs, ys, Xv, yva = load_xy()
    row = fit_row(kernel, C, Xs, ys, Xv, yva)
    ensure_dir(RUN_DIR)
    target = cell_file(kernel, C)
    with tempfile.NamedTemporaryFile(mode="w", dir=RUN_DIR, delete=False) as f:
        json.dump(row, f)
    os.replace(f.name, target)
    print(f"kernel={kernel:7s} C={C:<6} train={row['train_acc']:.4f} "
          f"val={row['val_acc']:.4f} sv_frac={row['sv_fraction']:.4f} ({row['fit_seconds']}s)")


def sweep_rows(recorded):
    cells = {(kernel, C) for kernel in KERNELS for C in C_VALUES}
    if recorded:
        with SWEEP_FILE.open(newline="") as f:
            rows = [
                {"kernel": r["kernel"], "C": float(r["C"]),
                 "train_acc": float(r["train_acc"]), "val_acc": float(r["val_acc"]),
                 "n_support_vectors": int(r["n_support_vectors"]),
                 "sv_fraction": float(r["sv_fraction"]), "fit_seconds": float(r["fit_seconds"])}
                for r in csv.DictReader(f)
            ]
    else:
        missing = [cell_file(*cell).name for cell in sorted(cells) if not cell_file(*cell).is_file()]
        if missing:
            raise SystemExit("Run these sweep cells first: " + ", ".join(missing))
        rows = [json.loads(cell_file(*cell).read_text()) for cell in sorted(cells)]
    if len(rows) != len(cells) or {(r["kernel"], r["C"]) for r in rows} != cells:
        raise ValueError("Sweep must contain each specified kernel and C exactly once.")
    return rows


def finalize(recorded=False):
    rows = sweep_rows(recorded)
    best = max(rows, key=lambda r: (r["val_acc"], -r["C"], -r["sv_fraction"]))
    Xs, ys, Xv, yva, Xt, yte = load_xy(include_test=True)
    print(f"Selected by validation: {best['kernel']}, C={best['C']:g}", flush=True)
    clf = make_svm(best["kernel"], best["C"]).fit(Xs, ys)
    train_acc = accuracy_score(ys, clf.predict(Xs))
    val_acc = accuracy_score(yva, clf.predict(Xv))
    test_pred = clf.predict(Xt)
    test_acc = accuracy_score(yte, test_pred)
    if recorded and (abs(val_acc - best["val_acc"]) > 0.00005
                     or int(clf.n_support_.sum()) != best["n_support_vectors"]):
        raise ValueError("Recorded sweep does not match the current feature cache.")
    if not recorded:
        with SWEEP_FILE.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    joblib.dump(clf, MODEL_FILE)
    np.savez(TEST_FILE, predictions=test_pred, labels=yte)
    selection = {
        "representation": load_json_q4rep(),
        "sweep": rows,
        "selection_rule": "highest validation accuracy; ties -> lower C, then lower SV fraction",
        "selected": best,
        "final": {
            "kernel": best["kernel"], "C": best["C"],
            "train_acc": train_acc, "val_acc": val_acc,
            "test_acc": test_acc,
            "n_support_vectors": int(clf.n_support_.sum()),
            "sv_fraction": float(clf.n_support_.sum() / len(Xs)),
        },
    }
    save_json(selection, Q4DIR / "q4a_summary.json")
    print(f"FINAL selected SVM test_acc = {test_acc:.4f}")


def load_json_q4rep():
    with open(Q4DIR / "q4_representation.json") as f:
        return json.load(f)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=KERNELS)
    ap.add_argument("--C", type=float, choices=C_VALUES)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--finalize-only-recorded", action="store_true")
    args = ap.parse_args()
    ensure_dir(Q4DIR)
    if args.finalize or args.finalize_only_recorded:
        if args.kernel is not None or args.C is not None:
            ap.error("finalize modes do not accept --kernel or --C")
        finalize(recorded=args.finalize_only_recorded)
    else:
        if args.kernel is None or args.C is None:
            ap.error("provide --kernel and --C, or select a finalize mode")
        task(args.kernel, args.C)


if __name__ == "__main__":
    main()

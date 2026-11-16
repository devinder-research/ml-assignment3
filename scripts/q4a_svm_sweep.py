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

Execution model: every (kernel, C) fit is an independent task executed by
`--kernel K --C c`; each task appends its row to results/q4/sweep_rows.jsonl.
`--finalize` assembles sweep.csv, applies the SELECTION RULE (highest
validation accuracy; ties -> lower C, then lower SV fraction), and evaluates
the selected model on the official test set ONCE. The linear-kernel rows and
the RBF C=0.01 row were computed with this exact code (sequentially, seed
260004) and are seeded verbatim from the completed run's log
(see results/q4/sweep_seeded.json) because rerunning linear C=100 alone cost
~1 core-hour; all remaining settings are computed by the task mode.

For EVERY setting we record training accuracy, validation accuracy, the
fraction of training points that become support vectors, and fit seconds.
The official test set is evaluated once, after the selection is fixed.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils import (DATA_DIR, RESULTS_DIR, save_json, SEED, ensure_dir)

C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
KERNELS = ["linear", "rbf", "poly"]
POLY_DEGREE = 3
Q4DIR = RESULTS_DIR / "q4"
ROWS_FILE = Q4DIR / "sweep_rows.jsonl"
SEEDED_FILE = Q4DIR / "sweep_seeded.json"


def load_xy():
    ftr = np.load(DATA_DIR / "q4_features_train.npy")
    fva = np.load(DATA_DIR / "q4_features_val.npy")
    fte = np.load(DATA_DIR / "q4_features_test.npy")
    ytr = np.load(DATA_DIR / "q4_labels_train.npy")
    yva = np.load(DATA_DIR / "q4_labels_val.npy")
    yte = np.load(DATA_DIR / "q4_labels_test.npy")
    sub = np.load(RESULTS_DIR / "q4" / "svm_subset_seed260004.npz")["subset_idx"]
    scaler = StandardScaler().fit(ftr[sub])
    return (scaler.transform(ftr[sub]), ytr[sub],
            scaler.transform(fva), yva, scaler.transform(fte), yte)


def fit_row(kernel, C, Xs, ys, Xv, yva):
    kwargs = dict(C=C, kernel=kernel, cache_size=1000)
    if kernel == "poly":
        kwargs.update(degree=POLY_DEGREE)
    t0 = time.time()
    clf = SVC(**kwargs)
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
    Xs, ys, Xv, yva, _, _ = load_xy()
    row = fit_row(kernel, C, Xs, ys, Xv, yva)
    ensure_dir(Q4DIR)
    with open(ROWS_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"kernel={kernel:7s} C={C:<6} train={row['train_acc']:.4f} "
          f"val={row['val_acc']:.4f} sv_frac={row['sv_fraction']:.4f} ({row['fit_seconds']}s)")


def finalize():
    ensure_dir(Q4DIR)
    rows = []
    if SEEDED_FILE.exists():
        rows.extend(json.load(open(SEEDED_FILE)))
    if ROWS_FILE.exists():
        with open(ROWS_FILE) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    # de-duplicate, preferring the freshly computed row
    uniq = {}
    for r in rows:
        uniq[(r["kernel"], r["C"])] = r
    rows = [uniq[k] for k in sorted(uniq, key=lambda k: (k[0], k[1]))]
    assert len(rows) == len(KERNELS) * len(C_VALUES), (
        f"expected 15 rows, have {len(rows)}: {[(r['kernel'], r['C']) for r in rows]}")

    import csv
    with open(Q4DIR / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    Xs, ys, Xv, yva, Xt, yte = load_xy()

    best = max(rows, key=lambda r: (r["val_acc"], -r["C"], -r["sv_fraction"]))
    print(f"\nSELECTED by validation: kernel={best['kernel']} C={best['C']} "
          f"val_acc={best['val_acc']:.4f}")

    # ---- single official test evaluation of the selected model
    kwargs = dict(C=best["C"], kernel=best["kernel"])
    if best["kernel"] == "poly":
        kwargs.update(degree=POLY_DEGREE)
    t0 = time.time()
    clf = SVC(**kwargs).fit(Xs, ys)
    train_acc = accuracy_score(ys, clf.predict(Xs))
    val_acc = accuracy_score(yva, clf.predict(Xv))
    test_pred = clf.predict(Xt)
    test_acc = accuracy_score(yte, test_pred)
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
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    ensure_dir(Q4DIR)
    if args.finalize:
        finalize()
    else:
        assert args.kernel in KERNELS and args.C in C_VALUES
        task(args.kernel, args.C)


if __name__ == "__main__":
    main()

"""Q4(b): difficult class pairs for the selected SVM - support vectors & near-boundary test points.

Difficulty is measured ON THE TEST SET from the confusion matrix of the
selected SVM (the question explicitly asks for pairs difficult for the
selected SVM on the test set): we rank unordered class pairs by the total
bidirectional confusion mass (n(i->j) + n(j->i)) and take the three largest.

For each pair (i, j), using one-vs-rest decision scores f_k of the selected SVM:
  * pair margin: d(x) = f_i(x) - f_j(x);  boundary-closest <=> |d(x)| smallest;
  * training side: support vectors (union of the one-vs-rest support masks)
    with the smallest |d| are shown;
  * test side: official test examples with the smallest |d| are shown;
  * we record the margin, true/predicted labels and relate them to errors,
    plus feature-space geometry (centroid distance vs within-class radius).

Outputs: results/q4/q4b_summary.json and figs/q4_pair_<i>_<j>.png grids.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils import (DATA_DIR, RESULTS_DIR, CIFAR10_CLASSES, SEED,
                       ensure_dir, load_json, save_json)
from src.data import load_split_arrays, create_split

POLY_DEGREE = 3
N_SHOW = 6


def main():
    ftr = np.load(DATA_DIR / "q4_features_train.npy")
    fte = np.load(DATA_DIR / "q4_features_test.npy")
    ytr = np.load(DATA_DIR / "q4_labels_train.npy")
    yte = np.load(DATA_DIR / "q4_labels_test.npy")
    raw_train, _, raw_test, _ = load_split_arrays()
    tr_split_idx, _ = create_split()
    raw_train_split = raw_train[tr_split_idx]  # same 45k order as the split

    sub = np.load(RESULTS_DIR / "q4" / "svm_subset_seed260004.npz")["subset_idx"]
    q4a = load_json(RESULTS_DIR / "q4" / "q4a_summary.json")
    sel = q4a["selected"]
    print("selected:", sel["kernel"], "C=", sel["C"])

    scaler = StandardScaler().fit(ftr[sub])
    Xs, Xt = scaler.transform(ftr[sub]), scaler.transform(fte)
    ys = ytr[sub]

    kwargs = dict(C=sel["C"], kernel=sel["kernel"], random_state=SEED)
    if sel["kernel"] == "poly":
        kwargs["degree"] = POLY_DEGREE
    t0 = time.time()
    clf = SVC(**kwargs).fit(Xs, ys)
    print(f"refit selected SVM in {time.time() - t0:.1f}s")
    ypred = clf.predict(Xt)
    test_acc = float((ypred == yte).mean())
    cm = confusion_matrix(yte, ypred, labels=np.arange(10))

    pairs = []
    for i in range(10):
        for j in range(i + 1, 10):
            pairs.append(((i, j), int(cm[i, j] + cm[j, i])))
    pairs.sort(key=lambda p: -p[1])
    hard = pairs[:3]
    print("hardest pairs:", [((CIFAR10_CLASSES[i], CIFAR10_CLASSES[j]), m) for (i, j), m in hard])

    Df = clf.decision_function(Xt)     # (10000, 10) OVR scores
    Dtr = clf.decision_function(Xs)    # (9000, 10)
    sv_mask = np.zeros(len(Xs), dtype=bool)
    sv_mask[clf.support_] = True

    summary = {"test_acc": test_acc, "confusion": cm.tolist(),
               "sv_per_class": clf.n_support_.astype(int).tolist(), "pairs": []}

    def rows_for(idx, D, labels, kind, i, j):
        rows = []
        for t in idx:
            rows.append({
                "index": int(t),
                "true": CIFAR10_CLASSES[int(labels[t])],
                "pred": CIFAR10_CLASSES[int(np.argmax(D[t]))],
                "d_pair": float(D[t, i] - D[t, j]),
                "f_i": float(D[t, i]), "f_j": float(D[t, j]),
                "kind": kind,
            })
        return rows

    for (i, j), mass in hard:
        d_te = Df[:, i] - Df[:, j]
        d_tr = Dtr[:, i] - Dtr[:, j]
        te_idx = np.argsort(np.abs(d_te))[:N_SHOW]
        tr_pool = [t for t in np.argsort(np.abs(d_tr))[:200] if sv_mask[t]][:N_SHOW]
        tr_idx = np.array(tr_pool, dtype=int)

        te_rows = rows_for(te_idx, Df, yte, "test-near-boundary", i, j)
        tr_rows = rows_for(tr_idx, Dtr, ys, "train-support-vector-near-boundary", i, j)

        te_flags = ypred[te_idx] == yte[te_idx]
        n_err = int((~te_flags).sum())
        pair_mask = (yte == i) | (yte == j)
        pair_err_rate = float((ypred[pair_mask] != yte[pair_mask]).mean())

        ci, cj = Xs[ys == i].mean(0), Xs[ys == j].mean(0)
        centroid_dist = float(np.linalg.norm(ci - cj))
        within_i = float(np.sqrt(((Xs[ys == i] - ci) ** 2).sum(1).mean()))
        within_j = float(np.sqrt(((Xs[ys == j] - cj) ** 2).sum(1).mean()))

        summary["pairs"].append({
            "pair": [CIFAR10_CLASSES[i], CIFAR10_CLASSES[j]],
            "pair_indices": [int(i), int(j)],
            "confusion_mass_test": mass,
            "cm_ij": int(cm[i, j]), "cm_ji": int(cm[j, i]),
            "near_boundary_test_errors": n_err, "near_boundary_test_n": len(te_idx),
            "pair_error_rate_test": pair_err_rate,
            "overall_error_rate_test": 1.0 - test_acc,
            "centroid_distance": centroid_dist,
            "within_class_radius": [within_i, within_j],
            "test_examples": te_rows,
            "train_support_vectors": tr_rows,
        })

        fig, axes = plt.subplots(2, N_SHOW, figsize=(N_SHOW * 1.7, 4.2))
        for k in range(N_SHOW):
            axt, axb = axes[0, k], axes[1, k]
            if k < len(tr_idx):
                axt.imshow(np.clip(raw_train_split[sub][tr_idx[k]], 0, 1))
                r = tr_rows[k]
                axt.set_title(f"SV d={r['d_pair']:+.1e}\nt={r['true']}", fontsize=7)
            axt.axis("off")
            if k < len(te_idx):
                axb.imshow(np.clip(raw_test[te_idx[k]], 0, 1))
                r = te_rows[k]
                axb.set_title(f"test d={r['d_pair']:+.1e}\nt={r['true']},p={r['pred']}", fontsize=7)
            axb.axis("off")
        fig.suptitle(f"{CIFAR10_CLASSES[i]} vs {CIFAR10_CLASSES[j]}: top = train SVs, bottom = near-boundary test", fontsize=8)
        figs_dir = ensure_dir(RESULTS_DIR.parent / "figs")
        fig.savefig(figs_dir / f"q4_pair_{i}_{j}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    save_json(summary, RESULTS_DIR / "q4" / "q4b_summary.json")
    print("saved q4b_summary.json")


if __name__ == "__main__":
    main()

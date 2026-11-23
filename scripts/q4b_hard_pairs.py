"""Inspect the selected SVM's three most confused class pairs."""
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import pairwise_kernels

from q4a_svm_sweep import MODEL_FILE, TEST_FILE, load_xy
from src.data import create_split, get_raw_cifar10
from src.utils import CIFAR10_CLASSES, FIGS_DIR, RESULTS_DIR, ensure_dir, load_json, save_json

N_SHOW = 6
Q4DIR = RESULTS_DIR / "q4"


def pair_support(model, i, j):
    """Return positions and signed dual coefficients for the pair i vs j."""
    starts = np.r_[0, np.cumsum(model.n_support_)]
    positions_i = np.arange(starts[i], starts[i + 1])
    positions_j = np.arange(starts[j], starts[j + 1])
    active_i = positions_i[model.dual_coef_[j - 1, positions_i] != 0]
    active_j = positions_j[model.dual_coef_[i, positions_j] != 0]
    return (np.r_[active_i, active_j],
            np.r_[model.dual_coef_[j - 1, active_i], model.dual_coef_[i, active_j]])


def pair_scores(model, features, positions, coefficients, intercept):
    # Evaluate exactly the selected pair's binary decision function. Restrict
    # kernel computation to active SVs for this pair, as in libsvm's margin.
    kernels = pairwise_kernels(features.astype(np.float64, copy=False),
                               model.support_vectors_[positions],
                               metric=model.kernel, filter_params=True,
                               gamma=model._gamma, degree=model.degree, coef0=model.coef0)
    return kernels @ coefficients + intercept


def closest(scores):
    return np.argsort(np.abs(scores), kind="stable")[:N_SHOW]


def example_rows(indices, original_indices, scores, labels, predictions):
    return [
        {"index": int(original_indices[idx]),
         "true": CIFAR10_CLASSES[labels[idx]],
         "pred": CIFAR10_CLASSES[pred],
         "d_pair": float(score)}
        for idx, pred, score in zip(indices, predictions, scores)
    ]


def plot_pair(pair, train_rows, test_rows, train_images, test_images, path):
    fig, axes = plt.subplots(2, N_SHOW, figsize=(10.2, 3.7))
    for row, examples, images, label in (
        (axes[0], train_rows, train_images, "SV"),
        (axes[1], test_rows, test_images, "test"),
    ):
        for ax in row:
            ax.axis("off")
        for ax, example in zip(row, examples):
            ax.imshow(images[example["index"]])
            ax.set_title(f"{label} {example['d_pair']:+.2e}\n"
                         f"{example['true']} → {example['pred']}", fontsize=8)
    fig.suptitle(f"{pair[0]} vs {pair[1]} · Pairwise SVM scores", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.82,
                        hspace=0.55, wspace=0.12)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    Xs, ys, _, _, Xt, yte = load_xy(include_test=True)
    model = joblib.load(MODEL_FILE)  # Generated locally by Q4(a).
    with np.load(TEST_FILE) as test:
        ypred = test["predictions"]
        if not np.array_equal(test["labels"], yte):
            raise ValueError("Saved predictions do not align with the test split.")
    test_acc = float(np.mean(ypred == yte))
    if test_acc != load_json(Q4DIR / "q4a_summary.json")["final"]["test_acc"]:
        raise ValueError("Run Q4(a) finalization for the current features before Q4(b).")

    pairs = list(combinations(range(len(model.classes_)), 2))
    cm = confusion_matrix(yte, ypred, labels=model.classes_)
    hard_pairs = sorted(pairs, key=lambda pair: -(cm[pair] + cm[pair[::-1]]))[:3]
    train_images = get_raw_cifar10(train=True).data
    test_images = get_raw_cifar10(train=False).data
    train_idx, _ = create_split()
    with np.load(Q4DIR / "svm_subset_seed260004.npz") as subset:
        train_original = train_idx[subset["subset_idx"]]

    summary = {"test_acc": test_acc, "confusion": cm.tolist(),
               "sv_per_class": model.n_support_.tolist(), "pairs": []}
    ensure_dir(FIGS_DIR)
    for i, j in hard_pairs:
        intercept = model.intercept_[pairs.index((i, j))]
        positions, coefficients = pair_support(model, i, j)
        pair_sv = model.support_[positions]
        pair_mask = (yte == i) | (yte == j)
        pair_test = np.flatnonzero(pair_mask)
        d_test = pair_scores(model, Xt[pair_test], positions, coefficients, intercept)
        d_train = pair_scores(model, Xs[pair_sv], positions, coefficients, intercept)
        near_test, near_train = closest(d_test), closest(d_train)
        te_idx = pair_test[near_test]
        tr_idx = pair_sv[near_train]
        te_rows = example_rows(te_idx, np.arange(len(yte)), d_test[near_test], yte, ypred[te_idx])
        tr_rows = example_rows(tr_idx, train_original, d_train[near_train], ys, model.predict(Xs[tr_idx]))
        groups = [Xs[ys == cls] for cls in (i, j)]
        centers = [group.mean(axis=0) for group in groups]
        radii = [float(np.sqrt(np.square(g - c).sum(axis=1).mean()))
                 for g, c in zip(groups, centers)]
        pair = [CIFAR10_CLASSES[k] for k in (i, j)]
        summary["pairs"].append({
            "pair": pair, "pair_indices": [i, j],
            "confusion_mass_test": int(cm[i, j] + cm[j, i]),
            "cm_ij": int(cm[i, j]), "cm_ji": int(cm[j, i]),
            "pair_support_count": len(pair_sv),
            "near_boundary_test_errors": int(np.sum(ypred[te_idx] != yte[te_idx])),
            "near_boundary_test_n": len(te_idx),
            "pairwise_vote_errors": int(np.sum(np.where(d_test[near_test] > 0, i, j) != yte[te_idx])),
            "pair_error_rate_test": float(np.mean(ypred[pair_mask] != yte[pair_mask])),
            "overall_error_rate_test": 1 - test_acc,
            "centroid_distance": float(np.linalg.norm(centers[0] - centers[1])),
            "within_class_radius": radii,
            "test_examples": te_rows, "train_support_vectors": tr_rows,
        })
        print(f"Inspected {pair[0]} vs {pair[1]}", flush=True)
        plot_pair(pair, tr_rows, te_rows, train_images, test_images,
                  FIGS_DIR / f"q4_pair_{i}_{j}.png")
    save_json(summary, Q4DIR / "q4b_summary.json")


if __name__ == "__main__":
    main()

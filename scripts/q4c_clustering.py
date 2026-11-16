"""Q4(c): clustering the fixed representation (no labels during fitting).

Method and number of clusters are chosen with TRAINING + VALIDATION data only:
  * method: KMeans (Euclidean geometry on the standardized fixed features;
    simple, deterministic, and appropriate for compact, spherical clusters);
  * number of clusters k in {10, 20}: compared on a fixed random 1,500-sample
    subset of the 5,000 validation images using the silhouette score
    (higher is better). The official test set is NOT used for this choice.
  * fitting data for the final model: ALL 45,000 training representations
    (MiniBatchKMeans, batch 4096, n_init=4, seed 260004);
  * assignment: test representations are assigned to clusters WITHOUT labels;
    test labels are used only afterwards, for evaluation.

Evaluation on the official test set: ARI, NMI, cluster->class contingency
table, cluster purity, and per-class recall after majority-mapping each
cluster to its most frequent test class (reporting only). We then compare
these numbers with the Q4 SVM test results and identify at least one
class/pair where classification and clustering lead to different conclusions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

from src.utils import (DATA_DIR, RESULTS_DIR, CIFAR10_CLASSES, SEED, save_json,
                       ensure_dir, load_json)

K_CANDIDATES = [10, 20]
SIL_SUBSAMPLE = 1500
FIT_BATCH = 4096


def main():
    ftr = np.load(DATA_DIR / "q4_features_train.npy")
    fva = np.load(DATA_DIR / "q4_features_val.npy")
    fte = np.load(DATA_DIR / "q4_features_test.npy")
    ytr = np.load(DATA_DIR / "q4_labels_train.npy")
    yva = np.load(DATA_DIR / "q4_labels_val.npy")
    yte = np.load(DATA_DIR / "q4_labels_test.npy")

    scaler = StandardScaler().fit(ftr)   # clustering uses ALL training reps
    Xtr, Xva, Xte = scaler.transform(ftr), scaler.transform(fva), scaler.transform(fte)

    # ---- choose k using validation data only (silhouette on fixed val subsample)
    rng = np.random.RandomState(SEED)
    sil_idx = rng.choice(len(Xva), size=min(SIL_SUBSAMPLE, len(Xva)), replace=False)
    k_scores = {}
    for k in K_CANDIDATES:
        km = MiniBatchKMeans(n_clusters=k, batch_size=FIT_BATCH, n_init=4,
                             random_state=SEED)
        km.fit(Xtr)
        labels_va = km.predict(Xva[sil_idx])
        sil = silhouette_score(Xva[sil_idx], labels_va, random_state=SEED)
        k_scores[k] = float(sil)
        print(f"k={k}: val silhouette (subsample {len(sil_idx)}) = {sil:.4f}")

    k_best = max(k_scores, key=k_scores.get)
    print(f"chosen k by validation silhouette: {k_best}")

    # ---- choose the METHOD by validation silhouette as well
    method_candidates = {
        "minibatch-kmeans": MiniBatchKMeans(n_clusters=k_best, batch_size=FIT_BATCH,
                                            n_init=4, random_state=SEED),
        "kmeans": KMeans(n_clusters=k_best, n_init=10, random_state=SEED),
        "gmm-diag": GaussianMixture(n_components=k_best, covariance_type="diag",
                                    reg_covar=1e-4, random_state=SEED),
    }
    method_scores = {}
    for mname, model in method_candidates.items():
        model.fit(Xtr)
        pred_s = model.predict(Xva[sil_idx])
        method_scores[mname] = float(silhouette_score(Xva[sil_idx], pred_s,
                                                      random_state=SEED))
        print(f"method={mname:18s} val silhouette = {method_scores[mname]:+.4f}")

    method_best = max(method_scores, key=method_scores.get)
    print(f"chosen method by validation silhouette: {method_best}")

    if method_best == "minibatch-kmeans":
        km = MiniBatchKMeans(n_clusters=k_best, batch_size=FIT_BATCH,
                             n_init=8, random_state=SEED)
    elif method_best == "kmeans":
        km = KMeans(n_clusters=k_best, n_init=10, random_state=SEED)
    else:
        km = GaussianMixture(n_components=k_best, covariance_type="diag",
                             reg_covar=1e-4, random_state=SEED)
    km.fit(Xtr)
    test_clusters = km.predict(Xte)

    # ---- evaluation (labels used AFTER assignment only)
    ARI = adjusted_rand_score(yte, test_clusters)
    NMI = normalized_mutual_info_score(yte, test_clusters)
    AMI = adjusted_mutual_info_score(yte, test_clusters)

    contingency = np.zeros((k_best, 10), dtype=int)
    for c in range(10):
        contingency[:, c] = np.bincount(test_clusters[yte == c], minlength=k_best)
    purity = contingency.max(axis=1).sum() / contingency.sum()

    # majority-map clusters to test classes (reporting only)
    mapping = contingency.argmax(axis=1)
    mapped_pred = mapping[test_clusters]
    cluster_acc = float((mapped_pred == yte).mean())

    # per-class recall (after majority mapping) and cluster composition
    per_class_recall = {}
    for c in range(10):
        mask = yte == c
        per_class_recall[CIFAR10_CLASSES[c]] = float((mapped_pred[mask] == c).mean())

    # ---- comparison with SVM (from Q4b / q4a)
    q4a = load_json(RESULTS_DIR / "q4" / "q4a_summary.json")
    q4b_file = RESULTS_DIR / "q4" / "q4b_summary.json"
    q4b = load_json(q4b_file) if q4b_file.exists() else None
    svm_test_acc = q4a["final"]["test_acc"]

    # per-class recall of SVM for the same comparison
    svm_recall = None
    if q4b is not None:
        cm = np.array(q4b["confusion"])
        svm_recall = {CIFAR10_CLASSES[c]: float(cm[c, c] / cm[c].sum()) for c in range(10)}

    results = {
        "method": method_best,
        "method_candidates_val_silhouette": {m: v for m, v in method_scores.items()},
        "k_candidates": K_CANDIDATES,
        "val_silhouette": {str(k): v for k, v in k_scores.items()},
        "k_chosen_by_validation": k_best,
        "test_ARI": ARI, "test_NMI": NMI, "test_AMI": AMI,
        "test_cluster_purity": purity,
        "contingency_clusters_x_classes": contingency.tolist(),
        "cluster_to_class_map": [CIFAR10_CLASSES[m] for m in mapping],
        "test_acc_after_majority_mapping": cluster_acc,
        "per_class_recall_clustering": per_class_recall,
        "svm_test_acc": svm_test_acc,
        "per_class_recall_svm": svm_recall,
        "train_size": int(ftr.shape[0]), "test_size": int(fte.shape[0]),
    }
    save_json(results, RESULTS_DIR / "q4" / "q4c_summary.json")

    print("\n=== Q4(c) test-set evaluation ===")
    print(f"ARI={ARI:.4f}  NMI={NMI:.4f}  AMI={AMI:.4f}  purity={purity:.4f}")
    print(f"acc after majority mapping={cluster_acc:.4f}  (SVM test acc={svm_test_acc:.4f})")


if __name__ == "__main__":
    main()

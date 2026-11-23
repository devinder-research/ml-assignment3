"""Q4(a-c) common step: fixed CIFAR10 representation from Q3.

Fixed representation (choice recorded here, kept UNCHANGED for all of Q4):
  the bottleneck features of the VALIDATION-SELECTED clean convolutional
  autoencoder from Q3(a)  (selection: lowest validation MSE; see
  results/q3/q3a_summary.json). The encoder is frozen; features are the
  encoder's linear outputs before any nonlinearity.

Caches to data/ (gitignored):
  features_train.npy  (45000, B) float32  - fixed 45k train split
  features_val.npy    (5000, B)  float32  - fixed 5k validation split
  features_test.npy   (10000, B) float32  - official 10k test split
  svm_subset.npz      - indices (into the 45k train split) of the fixed
                        stratified 9,000-sample subset (seed 260004), chosen
                        BEFORE any SVM model selection; used for every SVM fit.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.data import split_arrays
from src.models import ConvAutoencoder
from src.utils import (DATA_DIR, RESULTS_DIR, get_device, save_json, set_seed,
                       SEED, load_json, ensure_dir)

SUBSET_SIZE = 9000
SUBSET_FILE = RESULTS_DIR / "q4" / "svm_subset_seed260004.npz"


def compute_features(model, images_t, device, batch=1024):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(images_t), batch):
            outs.append(model.encode(images_t[i:i + batch].to(device)).cpu().float().numpy())
    return np.concatenate(outs)


def main():
    device = get_device()
    print("device:", device)

    q3a = load_json(RESULTS_DIR / "q3" / "q3a_summary.json")
    sel_name = min(q3a, key=lambda k: q3a[k]["val_mse"])
    B = q3a[sel_name]["bottleneck"]
    print(f"fixed representation = frozen encoder of {sel_name} (B={B})")

    Xtr, ytr, Xval, yval, Xte, yte = split_arrays()
    Xtr_t = torch.from_numpy(np.transpose(Xtr, (0, 3, 1, 2))).float()
    Xval_t = torch.from_numpy(np.transpose(Xval, (0, 3, 1, 2))).float()
    Xte_t = torch.from_numpy(np.transpose(Xte, (0, 3, 1, 2))).float()

    set_seed(SEED)
    model = ConvAutoencoder(bottleneck=B).to(device)
    model.load_state_dict(torch.load(RESULTS_DIR / "q3" / sel_name / "best.pt",
                                     map_location=device)["state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)

    ftr = compute_features(model, Xtr_t, device)
    fva = compute_features(model, Xval_t, device)
    fte = compute_features(model, Xte_t, device)

    np.save(DATA_DIR / "q4_features_train.npy", ftr.astype(np.float32))
    np.save(DATA_DIR / "q4_features_val.npy", fva.astype(np.float32))
    np.save(DATA_DIR / "q4_features_test.npy", fte.astype(np.float32))
    np.save(DATA_DIR / "q4_labels_train.npy", ytr)
    np.save(DATA_DIR / "q4_labels_val.npy", yval)
    np.save(DATA_DIR / "q4_labels_test.npy", yte)

    # ---- fixed stratified subset (chosen before any model selection)
    ensure_dir(SUBSET_FILE.parent)
    if SUBSET_FILE.exists():
        sub = np.load(SUBSET_FILE)["subset_idx"]
        print("subset already exists:", sub.shape)
    else:
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(ytr))
        sub, _ = train_test_split(idx, train_size=SUBSET_SIZE, stratify=ytr,
                                  random_state=SEED)
        sub = np.sort(sub)
        np.savez(SUBSET_FILE, subset_idx=sub, seed=SEED, size=SUBSET_SIZE,
                 source="fixed stratified subset of the 45k train split, chosen before model selection",
                 class_counts=np.bincount(ytr[sub]))
        print("subset class counts:", np.bincount(ytr[sub]))

    # Sweep cells belong to this feature version. Clear stale generated rows
    # before another sweep can be finalized.
    sweep_dir = DATA_DIR / "q4_sweep"
    if sweep_dir.exists():
        for row in sweep_dir.glob("*.json"):
            row.unlink()
    for name in ("q4_selected.joblib", "q4_selected_test.npz"):
        (DATA_DIR / name).unlink(missing_ok=True)

    save_json({"representation": f"frozen bottleneck features of {sel_name}",
               "bottleneck": int(B),
               "subset_size": int(SUBSET_SIZE), "subset_seed": SEED,
               "subset_file": str(SUBSET_FILE.relative_to(RESULTS_DIR.parent))},
              RESULTS_DIR / "q4" / "q4_representation.json")


if __name__ == "__main__":
    main()

"""Q1(b): can the MLP fit examples that should not generalise?

We train the same high-capacity MLP (3072-1024-512-10, Dropout=0) on
  (i) the true labels of the fixed 45,000-image training split,
 (ii) i.i.d. random labels for the same 45,000 images, and
 (iii) i.i.d. random labels for a stratified 4,500-image subset
 (a memorisation probe well inside the model's capacity);
tracking TRAIN and VALIDATION metrics every epoch. Validation always uses
the TRUE labels. (i), (ii): 400 epochs, Adam lr 1e-3 (Adam was retained:
higher rates stall or oscillate on noise gradients); (iii): 300 epochs.

If useful structure were learnable from the random labels, validation accuracy
would exceed chance (10%); if the MLP only memorises, training accuracy rises
to ~1 while validation stays flat/degrades.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.data import get_tensor_split
from src.models import MLP, count_parameters
from src.train import train_classifier, evaluate
from src.utils import (RESULTS_DIR, get_device, save_json, set_seed, SEED, Timer)

EPOCHS = 400
BATCH = 256
LR = 1e-3


def main():
    device = get_device()
    print("device:", device)
    set_seed(SEED)

    xtr, ytr, xval, yval, xte, yte = get_tensor_split(normalize=True)
    ytr = ytr.numpy()
    val_loader = DataLoader(TensorDataset(xval, yval), batch_size=1024, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorDataset(xte, yte), batch_size=1024, shuffle=False, num_workers=0)

    rng = np.random.RandomState(SEED)
    conditions = {
        "true": torch.from_numpy(ytr),
        "random": torch.from_numpy(rng.randint(0, 10, size=len(ytr))),
    }
    # Third condition: a stratified 4,500-image subset with random labels.
    # Memorising 4,500 noise targets is well within the model's reach, so
    # interpolation of arbitrary labels is demonstrated decisively rather than
    # by extrapolation of the full-split trajectory.
    from sklearn.model_selection import train_test_split
    idx_small, _ = train_test_split(np.arange(len(ytr)), train_size=4500,
                                    stratify=ytr, random_state=SEED)
    idx_small = np.sort(idx_small)
    conditions["random_4k"] = (torch.from_numpy(rng.randint(0, 10, size=len(idx_small))), idx_small)

    summary = {}
    for name, entry in conditions.items():
        if isinstance(entry, tuple):
            labels, subset = entry
            xcond, ycond = xtr[torch.from_numpy(subset)], labels
            ep = 300
        else:
            xcond, ycond = xtr, entry
            ep = EPOCHS
        set_seed(SEED)
        loader = DataLoader(TensorDataset(xcond, ycond), batch_size=BATCH, shuffle=True,
                            num_workers=0, generator=torch.Generator().manual_seed(SEED))
        model = MLP(hidden=(1024, 512), dropout=0.0)
        nparams = count_parameters(model)
        out_dir = RESULTS_DIR / "q1" / f"q1b-{name}"
        t = Timer()
        train_classifier(model, loader, val_loader, device,
                         epochs=ep, lr=LR, weight_decay=0.0,
                         log_path=out_dir / "log.csv",
                         ckpt_path=out_dir / "best.pt")
        state = torch.load(out_dir / "best.pt", map_location=device)
        model.load_state_dict(state["state_dict"])
        test = evaluate(model, test_loader, device)
        summary[name] = {
            "name": f"q1b-{name}", "params": nparams, "epochs": ep,
            "batch_size": BATCH, "lr": LR, "optimizer": "Adam",
            "best_epoch": state["epoch"], "best_val_acc": state["val_acc"],
            "test_loss": test["loss"], "test_acc": test["acc"],
            "train_seconds": round(t.elapsed(), 1),
        }
        save_json(summary[name], out_dir / "metrics.json")

    save_json(summary, RESULTS_DIR / "q1" / "q1b_summary.json")


if __name__ == "__main__":
    main()

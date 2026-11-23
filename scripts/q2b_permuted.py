"""Q2(b): destroy spatial structure while keeping the exact same pixel values.

A fixed random permutation of the 3072 CIFAR10 coordinates (seed = 260004,
saved to split/pixel_permutation_seed260004.npy) is applied THE SAME WAY to
training, validation and test images. Both models are retrained from scratch
with the Q2(a) protocol.

Prior expectation (stated before training):
  * q2-mlp: essentially unaffected (dense layers are invariant to input
    coordinate relabeling up to re-learning of the same function);
  * q2-cnn: collapses to (or near) MLP level, because its locality /
    weight-sharing prior no longer matches the data.
The observed change tests the explanation proposed in Q2(a).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import make_classifiers_loaders, get_pixel_permutation
from src.models import CNN, MLP, count_parameters
from src.train import train_classifier, evaluate
from src.utils import (RESULTS_DIR, get_device, save_json, set_seed, SEED, Timer)

EPOCHS = 40
BATCH = 256
LR = 1e-3

CONFIGS = {
    "q2-mlp-permuted": lambda: MLP(hidden=(512, 256), dropout=0.0),
    "q2-cnn-permuted": lambda: CNN(),
}


def main():
    device = get_device()
    print("device:", device)
    perm = get_pixel_permutation()
    print("permutation loaded:", perm.shape, "first indices:", perm[:5].tolist())
    loaders = make_classifiers_loaders(permuted=True, batch_size=BATCH, seed=SEED)

    summary = {}
    for name, build in CONFIGS.items():
        set_seed(SEED)
        train_loader, val_loader, test_loader = loaders
        model = build()
        nparams = count_parameters(model)
        out_dir = RESULTS_DIR / "q2" / name
        t = Timer()
        train_classifier(model, train_loader, val_loader, device,
                         epochs=EPOCHS, lr=LR, weight_decay=0.0,
                         log_path=out_dir / "log.csv",
                         ckpt_path=out_dir / "best.pt")
        state = torch.load(out_dir / "best.pt", map_location=device)
        model.load_state_dict(state["state_dict"])
        test = evaluate(model, test_loader, device)

        summary[name] = {"name": name, "params": nparams,
                         "epochs": EPOCHS, "batch_size": BATCH, "lr": LR,
                         "best_epoch": state["epoch"], "best_val_acc": state["val_acc"],
                         "test_loss": test["loss"], "test_acc": test["acc"],
                         "train_seconds": round(t.elapsed(), 1)}
        save_json(summary[name], out_dir / "metrics.json")
        print(f"=== {name} | test_acc(permuted)={test['acc']:.4f} ===")

    save_json(summary, RESULTS_DIR / "q2" / "q2b_summary.json")


if __name__ == "__main__":
    main()

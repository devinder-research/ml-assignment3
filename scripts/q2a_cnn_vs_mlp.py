"""Q2(a): fair comparison of one CNN and one MLP on CIFAR10.

Fairness criterion (defined BEFORE training and before any test result is
seen):
  * identical data split (fixed seed 260004), no augmentation;
  * identical training budget: same epochs (40), optimizer (Adam lr 1e-3),
    batch size (256), loss, and no LR schedule;
  * approximately matched parameter budget: parameter counts within +/-20%
    (CNN happens to be the SMALLER model, so the comparison, if anything,
    favours the MLP).

Models:
  q2-mlp : 3072-512-256-10          (~1.71M params)
  q2-cnn : conv 32->64->128->256    (~1.44M params)

q2-cnn is reused in Q2(c) and as the fixed classifier of Q3(b).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import make_classifiers_loaders
from src.models import CNN, MLP, count_parameters
from src.train import train_classifier, evaluate
from src.utils import (RESULTS_DIR, get_device, save_json, set_seed, SEED, Timer)

EPOCHS = 40
BATCH = 256
LR = 1e-3

CONFIGS = {
    "q2-mlp": lambda: MLP(hidden=(512, 256), dropout=0.0),
    "q2-cnn": lambda: CNN(),
}


def main():
    device = get_device()
    print("device:", device)
    loaders = make_classifiers_loaders(permuted=False, batch_size=BATCH, seed=SEED)

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
        print(f"=== {name} | params={nparams:,} | best_val={state['val_acc']:.4f} | test_acc={test['acc']:.4f} ===")

    save_json(summary, RESULTS_DIR / "q2" / "q2a_summary.json")


if __name__ == "__main__":
    main()

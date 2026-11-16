"""Q1(a): why might an MLP perform poorly on CIFAR10?

We test two competing hypotheses, keeping the models under a similar
parameter and training budget:

  H1 (chosen reason): MLPs lack spatial inductive bias, so they overfit the
  45,000 training images - seen as train accuracy >> validation accuracy,
  with validation accuracy leaving returns on the table.
  H2 (alternative): the MLP underfits - training itself stalls (optimization
  / insufficient effective capacity), so train accuracy also stays low.

Comparison (same data, same epochs, same optimizer, budget matched):
  mlp-base : MLP 3072-1024-512-10, no regularisation   (~3.7M params)
  mlp-reg  : identical architecture + Dropout(0.3) + weight_decay 1e-4
             (identical parameter count and training budget to mlp-base)
  mlp-small: MLP 3072-256-128-10, no regularisation      (~0.8M params)
             (capacity probe - smaller budget, reported separately)

Fairness criterion (declared before looking at results):
  same official split, no augmentation, same optimizer (Adam, lr 1e-3),
  same epochs (40) and batch size (256); parameter counts within the same
  order of magnitude, with mlp-base vs mlp-reg exactly matched.

Selected comparison (fair, matched budget): mlp-base vs mlp-reg.
Test accuracy is reported once, after all choices are fixed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import make_classifiers_loaders
from src.models import MLP, count_parameters
from src.train import train_classifier, evaluate
from src.utils import (RESULTS_DIR, get_device, save_json, set_seed, SEED, Timer)

EPOCHS = 40
BATCH = 256
LR = 1e-3

CONFIGS = {
    "mlp-base":  {"hidden": (1024, 512), "dropout": 0.0, "weight_decay": 0.0},
    "mlp-reg":   {"hidden": (1024, 512), "dropout": 0.3, "weight_decay": 1e-4},
    "mlp-small": {"hidden": (256, 128),  "dropout": 0.0, "weight_decay": 0.0},
}


def run(name, cfg, loaders, device):
    set_seed(SEED)
    train_loader, val_loader, test_loader = loaders
    model = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"])
    nparams = count_parameters(model)
    print(f"\n=== {name} | params={nparams:,} | cfg={cfg} ===")

    out_dir = RESULTS_DIR / "q1" / name
    t = Timer()
    train_classifier(model, train_loader, val_loader, device,
                     epochs=EPOCHS, lr=LR, weight_decay=cfg["weight_decay"],
                     log_path=out_dir / "log.csv",
                     ckpt_path=out_dir / "best.pt")

    # Final evaluation on the official test set (model selection used val only).
    state = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(state["state_dict"])
    test = evaluate(model, test_loader, device)
    metrics = {
        "name": name, "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "params": nparams, "epochs": EPOCHS, "batch_size": BATCH, "lr": LR,
        "best_epoch": state["epoch"], "best_val_acc": state["val_acc"],
        "test_loss": test["loss"], "test_acc": test["acc"],
        "train_seconds": round(t.elapsed(), 1),
    }
    save_json(metrics, out_dir / "metrics.json")
    print(f"=== {name} | best_val={state['val_acc']:.4f} | test_acc={test['acc']:.4f} ===")
    return metrics


def main():
    device = get_device()
    print("device:", device)
    loaders = make_classifiers_loaders(permuted=False, batch_size=BATCH, seed=SEED)
    results = {name: run(name, cfg, loaders, device) for name, cfg in CONFIGS.items()}
    save_json(results, RESULTS_DIR / "q1" / "q1a_summary.json")


if __name__ == "__main__":
    main()

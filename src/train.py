"""Training and evaluation helpers with per-epoch CSV logging."""
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .utils import ensure_dir


def read_log(path):
    with open(path) as f:
        return list(csv.DictReader(f))


@torch.no_grad()
def evaluate(model, loader, device, loss_fn=None):
    model.eval()
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    total_loss, total_n, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * y.numel()
        total_n += y.numel()
        correct += (logits.argmax(1) == y).sum().item()
    return {"loss": total_loss / total_n, "acc": correct / total_n}


@torch.no_grad()
def predict_probs(model, loader, device):
    """Return (probs, pred, true) numpy arrays for a classification loader."""
    model.eval()
    probs, pred, true = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        pred.append(logits.argmax(1).cpu().numpy())
        true.append(y.numpy())
    return np.concatenate(probs), np.concatenate(pred), np.concatenate(true)


def train_classifier(model, train_loader, val_loader, device, *,
                     epochs, lr=1e-3, weight_decay=0.0, log_path,
                     ckpt_path=None, optimizer=None, noisy_loader=None):
    """Standard classifier training with per-epoch train/val logging.

    Saves the checkpoint with the best VALIDATION accuracy (all architecture /
    hyperparameter selection uses validation only). Pass `optimizer=` to
    override the default Adam(lr, weight_decay).
    """
    model = model.to(device)
    log_path = Path(log_path)
    ensure_dir(log_path.parent)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "elapsed_s"]
    if noisy_loader is not None:
        fieldnames.append("noisy_target_acc")
    best_val, t0 = -1.0, time.time()
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss, train_n, train_correct = 0.0, 0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * y.numel()
                train_n += y.numel()
                train_correct += (logits.argmax(1) == y).sum().item()

            val = evaluate(model, val_loader, device, loss_fn)
            row = {"epoch": epoch, "train_loss": train_loss / train_n,
                   "train_acc": train_correct / train_n,
                   "val_loss": val["loss"], "val_acc": val["acc"],
                   "lr": optimizer.param_groups[0]["lr"],
                   "elapsed_s": round(time.time() - t0, 1)}
            if noisy_loader is not None:
                row["noisy_target_acc"] = evaluate(model, noisy_loader, device, loss_fn)["acc"]
            w.writerow(row)
            f.flush()
            print(f"epoch {epoch:3d} | train_loss {row['train_loss']:.4f} "
                  f"train_acc {row['train_acc']:.4f} | "
                  f"val_loss {row['val_loss']:.4f} val_acc {row['val_acc']:.4f}")
            if ckpt_path is not None and val["acc"] > best_val:
                best_val = val["acc"]
                ensure_dir(Path(ckpt_path).parent)
                torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                            "val_acc": val["acc"]}, ckpt_path)
    return model




"""Q3(a): three convolutional autoencoders differing ONLY in bottleneck capacity.

Architecture (identical elsewhere):
  conv(3->32,s2) -> conv(32->64,s2) -> conv(64->128,s2) -> fc(2048 -> B)
  fc(B -> 2048) -> deconv(128->64) -> deconv(64->32) -> deconv(32->3) -> sigmoid

Bottlenecks (clearly separated): {32, 256, 1024}.
Training: Adam lr 1e-3, batch 128, BCE-free MSE loss on [0,1] images, 40 epochs,
no augmentation, early stopping not used - identical budget for the three.
Selection criterion (reconstruction error only): lowest VALIDATION MSE.
Official test set: used once, for the selected configuration and all reported
test numbers.
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.data import ae_loaders
from src.models import ConvAutoencoder, count_parameters
from src.utils import (RESULTS_DIR, get_device, save_json, set_seed, SEED, Timer, ensure_dir)
from src.train import read_log

EPOCHS = 40
BATCH = 128
LR = 1e-3
BOTTLENECKS = [32, 256, 1024]


def run_ae(B, loaders, device):
    set_seed(SEED)
    train_loader, val_loader, test_loader = loaders
    model = ConvAutoencoder(bottleneck=B).to(device)
    nparams = count_parameters(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    out_dir = RESULTS_DIR / "q3" / f"ae-b{B}"
    ensure_dir(out_dir)
    log_path = out_dir / "log.csv"

    best_val = float("inf")
    t0 = time.time()
    fieldnames = ["epoch", "train_mse", "val_mse", "elapsed_s"]
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            tot, n = 0.0, 0
            for x in train_loader:
                x = x.to(device)
                opt.zero_grad(set_to_none=True)
                recon = model(x)
                loss = loss_fn(recon, x)
                loss.backward()
                opt.step()
                tot += loss.item() * x.numel()
                n += x.numel()
            val_mse = evaluate_ae(model, val_loader, device)
            row = {"epoch": epoch, "train_mse": tot / n, "val_mse": val_mse,
                   "elapsed_s": round(time.time() - t0, 1)}
            w.writerow(row)
            f.flush()
            print(f"B={B} epoch {epoch:3d} train_mse {row['train_mse']:.5f} val_mse {val_mse:.5f}")
            if val_mse < best_val:
                best_val = val_mse
                torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                            "val_mse": val_mse}, out_dir / "best.pt")
    return model, nparams, best_val, out_dir


@torch.no_grad()
def evaluate_ae(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    tot, n = 0.0, 0
    for x in loader:
        x = x.to(device)
        loss = loss_fn(model(x), x)
        tot += loss.item() * x.numel()
        n += x.numel()
    return tot / n


def extract_recon_batch(model, loader, device, n=12):
    model.eval()
    batch = next(iter(loader))[:n].to(device)
    with torch.no_grad():
        recon = model(batch)
    return batch.cpu(), recon.cpu()


def main():
    device = get_device()
    print("device:", device)
    loaders = ae_loaders(batch_size=BATCH, num_workers=0, seed=SEED)
    train_loader, val_loader, test_loader = loaders

    summary = {}
    for B in BOTTLENECKS:
        model, nparams, best_val, out_dir = run_ae(B, loaders, device)
        # official test reconstruction error (reported once; selection already fixed)
        state = torch.load(out_dir / "best.pt", map_location=device)
        model.load_state_dict(state["state_dict"])
        test_mse = evaluate_ae(model, test_loader, device)
        summary[f"ae-b{B}"] = {
            "bottleneck": B, "params": nparams, "epochs": EPOCHS,
            "batch_size": BATCH, "lr": LR,
            "best_epoch": state["epoch"], "val_mse": state["val_mse"],
            "test_mse": test_mse,
        }
        save_json(summary[f"ae-b{B}"], out_dir / "metrics.json")
        print(f"=== ae-b{B} | params={nparams:,} | val_mse={state['val_mse']:.5f} | test_mse={test_mse:.5f} ===")

    save_json(summary, RESULTS_DIR / "q3" / "q3a_summary.json")


if __name__ == "__main__":
    main()

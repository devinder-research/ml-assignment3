"""Q3(c): class information in the learned representations WITHOUT fine-tuning.

For every autoencoder trained in Q3(a) and Q3(b) (q3a clean AEs with
bottlenecks 32/256/1024, and the DAE), the ENCODER IS FROZEN and a linear
probe (Logistic-regression-style Linear(B -> 10), Adam lr 1e-3, 30 epochs)
is trained on the 45,000 training representations of CLEAN images.
We report validation and official-test probe accuracy.
The DAE is additionally probed on NOISY test images (its native input domain).

We then compare probe accuracy vs reconstruction MSE (validation and test)
and decide whether 'reconstruction quality' and 'class information' agree.
Official test labels are used here ONLY for the final probe evaluation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np

from q3b_denoising import evaluate_ae_mse
from src.data import split_arrays
from src.models import ConvAutoencoder
from src.utils import (RESULTS_DIR, get_device, save_json, set_seed, SEED, load_json)

PROBE_EPOCHS = 30
PROBE_LR = 1e-3
BATCH = 1024


@torch.no_grad()
def extract_features(model, images_t, device, batch=1024):
    model.eval()
    feats = []
    for i in range(0, len(images_t), batch):
        feats.append(model.encode(images_t[i:i + batch].to(device)).cpu())
    return torch.cat(feats)


def train_probe(Xtr_f, ytr, Xva_f, yva, device):
    set_seed(SEED)
    head = nn.Linear(Xtr_f.shape[1], 10).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=PROBE_LR)
    loss_fn = nn.CrossEntropyLoss()
    n = len(Xtr_f)
    Xa = Xtr_f.to(device)
    ya = torch.from_numpy(ytr).to(device)
    for epoch in range(1, PROBE_EPOCHS + 1):
        head.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED + epoch))
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(head(Xa[idx]), ya[idx])
            loss.backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        va_acc = (head(Xva_f.to(device)).argmax(1) == torch.from_numpy(yva).to(device)).float().mean().item()
    return head, va_acc


def probe_acc(head, Xf, y, device):
    head.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for i in range(0, len(Xf), 1024):
            correct += (head(Xf[i:i + 1024].to(device)).argmax(1) == torch.from_numpy(y[i:i + 1024]).to(device)).sum().item()
            n += len(Xf[i:i + 1024])
    return correct / n


def main():
    device = get_device()
    print("device:", device)

    Xtr, ytr, Xval, yval, Xte, yte = split_arrays()
    Xtr_t = torch.from_numpy(np.transpose(Xtr, (0, 3, 1, 2))).float()
    Xval_t = torch.from_numpy(np.transpose(Xval, (0, 3, 1, 2))).float()
    Xte_t = torch.from_numpy(np.transpose(Xte, (0, 3, 1, 2))).float()

    g = torch.Generator(device="cpu").manual_seed(SEED)
    noisy_te = ((Xte_t + torch.randn(Xte_t.shape, generator=g) * 0.25).clamp(0.0, 1.0))

    q3a = load_json(RESULTS_DIR / "q3" / "q3a_summary.json")
    q3b = load_json(RESULTS_DIR / "q3" / "q3b_summary.json")
    B_dae = q3b["dae_bottleneck"]

    specs = []
    for name, info in sorted(q3a.items()):  # ae-b32, ae-b256, ae-b1024
        ckpt = RESULTS_DIR / "q3" / name / "best.pt"
        specs.append((name, info["bottleneck"], ckpt))
    specs.append((f"dae-b{B_dae}", B_dae, RESULTS_DIR / "q3" / f"dae-b{B_dae}" / "best.pt"))

    results = {}
    for name, B, ckpt in specs:
        set_seed(SEED)
        model = ConvAutoencoder(bottleneck=B).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["state_dict"])
        for p in model.parameters():
            p.requires_grad_(False)

        Ftr = extract_features(model, Xtr_t, device)
        Fva = extract_features(model, Xval_t, device)
        Fte = extract_features(model, Xte_t, device)

        head, val_acc = train_probe(Ftr, ytr, Fva, yval, device)
        test_acc_clean = probe_acc(head, Fte, yte, device)

        entry = {
            "bottleneck": B,
            "frozen_encoder": True,
            "probe_epochs": PROBE_EPOCHS,
            "probe_lr": PROBE_LR,
            "probe_val_acc": val_acc,
            "probe_test_acc_clean": test_acc_clean,
        }
        # DAE also probed on noisy test images (its native corruption domain)
        if name.startswith("dae"):
            Fn = extract_features(model, noisy_te, device)
            entry["probe_test_acc_noisy"] = probe_acc(head, Fn, yte, device)
            entry["noisy_probe_seed"] = SEED
            entry["noisy_input_test_mse"] = q3b["dae_test_mse_on_noisy"]
            entry["val_mse"] = evaluate_ae_mse(model, Xval_t, Xval_t, device)
            entry["test_mse"] = evaluate_ae_mse(model, Xte_t, Xte_t, device)
        results[name] = entry
        save_json(entry, RESULTS_DIR / "q3" / name / "probe_metrics.json")
        print(f"{name}: val={val_acc:.4f} test(clean)={test_acc_clean:.4f}" +
              (f" test(noisy)={entry['probe_test_acc_noisy']:.4f}" if "probe_test_acc_noisy" in entry else ""))

    # attach reconstruction quality for the comparison
    for k in results:
        if k.startswith("ae"):
            m = load_json(RESULTS_DIR / "q3" / k / "metrics.json")
            results[k]["val_mse"] = m["val_mse"]
            results[k]["test_mse"] = m["test_mse"]

    save_json(results, RESULTS_DIR / "q3" / "q3c_summary.json")
    print("\n=== Q3(c) summary ===")
    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":
    main()

"""Q3(b): denoising autoencoder vs clean autoencoder for classification of corrupted images.

Corruption (chosen because reconstruction can plausibly undo it):
  additive Gaussian noise, sigma = 0.25, added to [0,1] images, clipped to [0,1].
  Applied IDENTICALLY at train (DAE inputs), validation and test time.
  Class labels are NOT used by either autoencoder.

Protocol:
  * Selected clean AE = the one with the LOWEST VALIDATION MSE in Q3(a)
    (read here from results/q3/q3a_summary.json; test was not used to choose).
  * DAE: same architecture as the selected clean AE, trained with noisy inputs
    and clean targets, same training budget as Q3(a).
  * Fixed CNN classifier = the Q2(a) CNN, trained on clean labelled data.
    NOT fine-tuned for any condition here.
  * Test-set conditions (10,000 official test images, one noise draw, seed 260004):
      1. clean test images (reference)                       [reused from Q2(a)]
      2. noisy test images, classified directly
      3. noisy test images -> clean AE -> reconstructions classified
      4. noisy test images -> DAE -> reconstructions classified
  * Reconstruction MSE of both autoencoders on the SAME noisy test inputs.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np

from src.data import split_arrays, normalize_batch
from src.models import CNN, ConvAutoencoder, count_parameters
from src.utils import (RESULTS_DIR, get_device, save_json, set_seed, SEED,
                       Timer, ensure_dir, load_json)

NOISE_SIGMA = 0.25
EPOCHS = 40
BATCH = 128
LR = 1e-3


def add_noise(x, sigma, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (x + torch.randn(x.shape, generator=g) * sigma).clamp(0.0, 1.0)


def evaluate_ae_mse(model, noisy, clean, device, batch=512):
    model.eval()
    tot, n = 0.0, 0
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for i in range(0, len(clean), batch):
            xc = noisy[i:i + batch].to(device)
            y = clean[i:i + batch].to(device)
            out = model(xc)
            tot += loss_fn(out, y).item() * y.numel()
            n += y.numel()
    return tot / n


def reconstructed_tensor(model, x, device, batch=512):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            outs.append(model(x[i:i + batch].to(device)).cpu())
    return torch.cat(outs)


def cnn_acc_on(model, images, labels, device, batch=512):
    """images: (N,3,32,32) [0,1] on CPU. Normalizes like the classifier was trained."""
    correct, n = 0, 0
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = normalize_batch(images[i:i + batch].to(device))
            y = labels[i:i + batch].to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            n += y.numel()
    return correct / n


def main():
    device = get_device()
    print("device:", device)
    set_seed(SEED)

    # ---- select clean AE by validation MSE (Q3a)
    q3a = load_json(RESULTS_DIR / "q3" / "q3a_summary.json")
    sel_name = min(q3a, key=lambda k: q3a[k]["val_mse"])
    B_sel = q3a[sel_name]["bottleneck"]
    print(f"selected clean AE: {sel_name} (val_mse={q3a[sel_name]['val_mse']:.5f})")

    # ---- train DAE (same architecture as selected clean AE)
    Xtr, ytr, Xval, yval, Xte, yte = split_arrays()
    Xtr_t = torch.from_numpy(np.transpose(Xtr, (0, 3, 1, 2))).float()
    Xval_t = torch.from_numpy(np.transpose(Xval, (0, 3, 1, 2))).float()
    Xte_t = torch.from_numpy(np.transpose(Xte, (0, 3, 1, 2))).float()
    ytr_t = torch.from_numpy(ytr)
    yte_t = torch.from_numpy(yte)

    set_seed(SEED)
    dae = ConvAutoencoder(bottleneck=B_sel).to(device)
    nparams = count_parameters(dae)
    opt = torch.optim.Adam(dae.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    out_dir = RESULTS_DIR / "q3" / f"dae-b{B_sel}"
    ensure_dir(out_dir)
    import csv
    log_path = out_dir / "log.csv"
    best_val = float("inf")
    t0 = time.time()
    n_train = len(Xtr_t)
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_mse", "val_mse", "elapsed_s"])
        w.writeheader()
        for epoch in range(1, EPOCHS + 1):
            dae.train()
            perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(SEED + epoch))
            tot, n = 0.0, 0
            for i in range(0, n_train, BATCH):
                idx = perm[i:i + BATCH]
                x_clean = Xtr_t[idx]
                x_noisy = add_noise(x_clean, NOISE_SIGMA, seed=SEED * 1000 + epoch * 100 + i).to(device)
                y = x_clean.to(device)
                opt.zero_grad(set_to_none=True)
                out = dae(x_noisy)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
                tot += loss.item() * y.numel()
                n += y.numel()
            # validation: noisy inputs, clean targets (same corruption as train)
            with torch.no_grad():
                dae.eval()
                val_mse_sum, val_n = 0.0, 0
                for i in range(0, len(Xval_t), 512):
                    xc = Xval_t[i:i + 512]
                    xn = add_noise(xc, NOISE_SIGMA, seed=4242 + i).to(device)
                    yc = xc.to(device)
                    val_mse_sum += loss_fn(dae(xn), yc).item() * yc.numel()
                    val_n += yc.numel()
                val_mse = val_mse_sum / val_n
            row = {"epoch": epoch, "train_mse": tot / n, "val_mse": val_mse,
                   "elapsed_s": round(time.time() - t0, 1)}
            w.writerow(row)
            f.flush()
            print(f"epoch {epoch:3d} train_mse {row['train_mse']:.5f} val_mse {val_mse:.5f}")
            if val_mse < best_val:
                best_val = val_mse
                torch.save({"state_dict": dae.state_dict(), "epoch": epoch,
                            "val_mse": val_mse}, out_dir / "best.pt")

    # ---- evaluation protocol
    dae.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["state_dict"])
    clean_ae = ConvAutoencoder(bottleneck=B_sel).to(device)
    clean_ae.load_state_dict(torch.load(RESULTS_DIR / "q3" / sel_name / "best.pt",
                                        map_location=device)["state_dict"])

    noisy_test = add_noise(Xte_t, NOISE_SIGMA, seed=999)  # ONE fixed noise draw
    recon_clean = reconstructed_tensor(clean_ae, noisy_test, device)
    recon_dae = reconstructed_tensor(dae, noisy_test, device)

    mse_clean_ae = evaluate_ae_mse(clean_ae, noisy_test, Xte_t, device)
    mse_dae = evaluate_ae_mse(dae, noisy_test, Xte_t, device)

    cnn = CNN().to(device)
    cnn_state = torch.load(RESULTS_DIR / "q2" / "q2-cnn" / "best.pt", map_location=device)
    cnn.load_state_dict(cnn_state["state_dict"])
    cnn.eval()

    acc_clean = cnn_acc_on(cnn, Xte_t, yte_t, device)
    acc_noisy = cnn_acc_on(cnn, noisy_test, yte_t, device)
    acc_recon_clean = cnn_acc_on(cnn, recon_clean, yte_t, device)
    acc_recon_dae = cnn_acc_on(cnn, recon_dae, yte_t, device)

    metrics = {
        "noise_sigma": NOISE_SIGMA,
        "selected_clean_ae": sel_name,
        "clean_ae_bottleneck": B_sel,
        "dae_bottleneck": B_sel,
        "dae_params": nparams,
        "dae_best_val_mse": best_val,
        "clean_ae_test_mse_on_noisy": mse_clean_ae,
        "dae_test_mse_on_noisy": mse_dae,
        "classifier": "q2-cnn (trained on clean labelled data, NOT fine-tuned here)",
        "test_acc_clean": acc_clean,
        "test_acc_noisy_direct": acc_noisy,
        "test_acc_cleanae_recon": acc_recon_clean,
        "test_acc_dae_recon": acc_recon_dae,
        "epochs": EPOCHS, "batch_size": BATCH, "lr": LR,
    }
    save_json(metrics, RESULTS_DIR / "q3" / "q3b_summary.json")
    print("\n=== Q3(b) summary ===")
    for k, v in metrics.items():
        print(k, "=", v)


if __name__ == "__main__":
    main()

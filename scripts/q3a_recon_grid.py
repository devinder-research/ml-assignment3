"""Save a grid of (clean image, clean-AE reconstruction) for the report.

Uses the VALIDATION-SELECTED clean AE and the denoising AE, on clean and on
noisy test images. Writes figs/q3a_recon_grid.png and figs/q3b_recon_grid.png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data import load_split_arrays
from src.models import ConvAutoencoder
from src.utils import RESULTS_DIR, FIGS_DIR, get_device, load_json, ensure_dir, SEED


def grid(images, recons, titles, path):
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.15, 2.45))
    for i in range(n):
        axes[0, i].imshow(images[i]); axes[0, i].axis("off")
        axes[1, i].imshow(recons[i]); axes[1, i].axis("off")
        axes[0, i].set_title(titles[i], fontsize=6)
    axes[0, 0].set_ylabel("target/clean")
    axes[1, 0].set_ylabel("recon")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    device = get_device()
    _, _, Xte, _ = load_split_arrays()
    q3a = load_json(RESULTS_DIR / "q3" / "q3a_summary.json")
    sel = min(q3a, key=lambda k: q3a[k]["val_mse"])
    B = q3a[sel]["bottleneck"]

    Xte_t = torch.from_numpy(np.transpose(Xte[:8], (0, 3, 1, 2))).float().to(device)

    ae = ConvAutoencoder(bottleneck=B).to(device)
    ae.load_state_dict(torch.load(RESULTS_DIR / "q3" / sel / "best.pt", map_location=device)["state_dict"])
    ae.eval()

    g = torch.Generator(device="cpu").manual_seed(999)
    noisy = ((Xte_t.cpu() + torch.randn(Xte_t.shape, generator=g) * 0.25).clamp(0, 1))

    with torch.no_grad():
        rec = ae(Xte_t).cpu()
    grid([Xte[i] for i in range(8)],
         [np.transpose(rec[i].numpy(), (1, 2, 0)) for i in range(8)],
         [f"#{i}" for i in range(8)],
         FIGS_DIR / "q3a_recon_grid.png")
    print("saved clean-AE grid")

    # DAE grid: only after Q3(b) has run (q3b_summary.json exists)
    q3b_path = RESULTS_DIR / "q3" / "q3b_summary.json"
    if q3b_path.exists():
        q3b = load_json(q3b_path)
        dae = ConvAutoencoder(bottleneck=q3b["dae_bottleneck"]).to(device)
        dae.load_state_dict(torch.load(RESULTS_DIR / "q3" / f"dae-b{q3b['dae_bottleneck']}" / "best.pt",
                                       map_location=device)["state_dict"])
        dae.eval()
        with torch.no_grad():
            rec_n = dae(noisy.to(device)).cpu()
        grid([np.clip(noisy[i].numpy(), 0, 1).transpose(1, 2, 0) for i in range(8)],
             [np.transpose(rec_n[i].numpy(), (1, 2, 0)) for i in range(8)],
             [f"#{i}" for i in range(8)],
             FIGS_DIR / "q3b_recon_grid.png")
        print("saved DAE grid")
    else:
        print("q3b_summary.json not found; DAE grid will be produced on a second run (after Q3b)")


if __name__ == "__main__":
    main()

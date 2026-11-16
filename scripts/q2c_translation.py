"""Q2(c): translation robustness experiment (distinguishing two explanations).

Candidate explanations for the CNN's advantage over the MLP on CIFAR10:
  E1 (translation invariance): the CNN's weight sharing builds an
      approximate shift-invariant detector, so object location is largely
      irrelevant.
  E2 (weight sharing / local features): the CNN needs fewer effective
      parameters and learns reusable LOCAL parts, but would not be any more
      robust to absolute object displacement than a sufficiently large MLP.

Both explanations predict the CNN uses space efficiently; they DIFFER on how
performance changes when the object's absolute position moves. We sweep
zero-padded test-time translations of the CLEAN test images by k in
{0, 1, 2, 4, 8} pixels along each of the four directions and report, for the
Q2(a) models (no retraining):
  * accuracy (4-direction mean),
  * accuracy per direction,
  * fraction of samples that were CORRECT at k=0 but become INCORRECT at k
    (flip-to-error rate) - the error analysis,
  * mean softmax confidence of correct and of incorrect predictions,
  * mean predictive entropy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import get_tensor_split, shift_tensor
from src.models import CNN, MLP
from src.utils import (RESULTS_DIR, get_device, save_json, set_seed, SEED)

KS = [0, 1, 2, 4, 8]
DIRECTIONS = [("+x", 1, 0), ("-x", -1, 0), ("+y", 0, 1), ("-y", 0, -1)]


@torch.no_grad()
def eval_shifted(model, loader, device, dx, dy):
    model.eval()
    conf_correct, conf_wrong, ent = [], [], []
    correct, n = 0, 0
    preds, trues = [], []
    for x, y in loader:
        x = shift_tensor(x, dx, dy).to(device)
        y = y.to(device)
        p = torch.softmax(model(x), dim=1)
        conf, pred = p.max(1)
        ent_vec = -(p * (p + 1e-12).log()).sum(1)
        ok = pred == y
        correct += ok.sum().item()
        n += y.numel()
        conf_correct += conf[ok].cpu().tolist()
        conf_wrong += conf[~ok].cpu().tolist()
        ent += ent_vec.cpu().tolist()
        preds.append(pred.cpu())
        trues.append(y.cpu())
    return {
        "acc": correct / n,
        "conf_correct": float(np.mean(conf_correct)) if conf_correct else None,
        "conf_wrong": float(np.mean(conf_wrong)) if conf_wrong else None,
        "mean_entropy": float(np.mean(ent)),
        "pred": np.concatenate(preds),
        "true": np.concatenate(trues),
        "n": n,
    }


def main():
    device = get_device()
    print("device:", device)
    set_seed(SEED)
    _, _, _, _, xte, yte = get_tensor_split(normalize=True)
    test_loader = DataLoader(TensorDataset(xte, yte), batch_size=1024,
                             shuffle=False, num_workers=0)

    models = {
        "q2-mlp": (MLP(hidden=(512, 256), dropout=0.0), RESULTS_DIR / "q2" / "q2-mlp" / "best.pt"),
        "q2-cnn": (CNN(), RESULTS_DIR / "q2" / "q2-cnn" / "best.pt"),
    }

    results = {}
    for name, (model, ckpt) in models.items():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["state_dict"])
        model = model.to(device)

        base = eval_shifted(model, test_loader, device, 0, 0)
        clean_correct = base["pred"] == base["true"]

        per_k = {}
        for k in KS:
            if k == 0:
                per_k["0"] = {"acc": base["acc"], "directions": {"none": base["acc"]},
                              "conf_correct": base["conf_correct"],
                              "conf_wrong": base["conf_wrong"],
                              "mean_entropy": base["mean_entropy"],
                              "flip_to_error": 0.0,
                              "err_rate": 1.0 - base["acc"]}
                print(f"{name} k=0  acc={base['acc']:.4f}")
                continue
            accs, dir_accs, cc, cw, ent, flips = [], {}, [], [], [], []
            for dname, sgnx, sgny in DIRECTIONS:
                r = eval_shifted(model, test_loader, device, sgnx * k, sgny * k)
                dir_accs[dname] = r["acc"]
                accs.append(r["acc"])
                cc.append(r["conf_correct"]); cw.append(r["conf_wrong"]); ent.append(r["mean_entropy"])
                newly_wrong = (r["pred"] != r["true"]) & clean_correct
                flips.append(newly_wrong.mean())
                print(f"{name} k={k} {dname}  acc={r['acc']:.4f} flip_to_error={newly_wrong.mean():.4f}")
            per_k[str(k)] = {
                "acc": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
                "directions": dir_accs,
                "conf_correct": float(np.mean([c for c in cc if c is not None])),
                "conf_wrong": float(np.mean([c for c in cw if c is not None])),
                "mean_entropy": float(np.mean(ent)),
                "flip_to_error": float(np.mean(flips)),
                "err_rate": float(1.0 - np.mean(accs)),
            }
        results[name] = per_k
        save_json(per_k, RESULTS_DIR / "q2" / f"{name}_translation.json")

    save_json(results, RESULTS_DIR / "q2" / "q2c_summary.json")


if __name__ == "__main__":
    main()

"""Generate all report figures from the logs/metrics produced by the experiments.

Reads results/*/log.csv and results/*/metrics.json, writes figs/*.png.
Style: colorblind-friendly palette, 300 dpi, sized for a 6.5cm two-column figure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

from src.utils import RESULTS_DIR, FIGS_DIR, ensure_dir

mpl.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_BLUE, C_RED, C_GREEN, C_ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"

ensure_dir(FIGS_DIR)


def read_log(name):
    with open(RESULTS_DIR / name / "log.csv") as f:
        return list(csv.DictReader(f))


def col(rows, key, cast=float):
    return [cast(r[key]) for r in rows]


# ---------------- Q1(a): train/val curves ----------------
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3), sharey=False)
for ax, name, label in [(axes[0], "q1/mlp-base", "MLP-base (no reg)"),
                        (axes[1], "q1/mlp-reg", "MLP-reg (dropout+wd)"),
                        (axes[2], "q1/mlp-small", "MLP-small")]:
    r = read_log(name)
    ep = col(r, "epoch", int)
    ax.plot(ep, col(r, "train_acc"), color=C_BLUE, label="train")
    ax.plot(ep, col(r, "val_acc"), color=C_RED, label="val")
    ax.set_title(label)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("accuracy")
axes[0].legend(loc="lower right")
fig.savefig(FIGS_DIR / "fig_q1a_curves.png")
plt.close(fig)

# ---------------- Q1(b): memorisation ----------------
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3))
for ax, name, label in [(axes[0], "q1/q1b-true", "true labels (45k)"),
                        (axes[1], "q1/q1b-random", "random labels (45k)"),
                        (axes[2], "q1/q1b-random_4k", "random labels (4.5k subset)")]:
    r = read_log(name)
    ep = col(r, "epoch", int)
    ax.plot(ep, col(r, "train_acc"), color=C_BLUE, label="train acc")
    ax.plot(ep, col(r, "val_acc"), color=C_RED, label="val acc")
    ax.set_xlabel("epoch")
    ax.set_title(label, fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("accuracy")
axes[0].legend(loc="center right", fontsize=6)
fig.savefig(FIGS_DIR / "fig_q1b_memorization.png")
plt.close(fig)

# Q1(b) gap figure (train minus val)
r_true = read_log("q1/q1b-true")
r_rand = read_log("q1/q1b-random")
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.plot(col(r_true, "epoch", int), [t - v for t, v in zip(col(r_true, "train_acc"), col(r_true, "val_acc"))],
        color=C_GREEN, label="true labels")
ax.plot(col(r_rand, "epoch", int), [t - v for t, v in zip(col(r_rand, "train_acc"), col(r_rand, "val_acc"))],
        color=C_ORANGE, label="random labels")
import os
if os.path.exists(RESULTS_DIR / "q1" / "q1b-random_4k" / "log.csv"):
    r_rand4 = read_log("q1/q1b-random_4k")
    ax.plot(col(r_rand4, "epoch", int), [t - v for t, v in zip(col(r_rand4, "train_acc"), col(r_rand4, "val_acc"))],
            color=C_RED, label="random labels (4.5k)")
ax.set_xlabel("epoch")
ax.set_ylabel("train acc - val acc")
ax.legend(fontsize=6)
ax.grid(alpha=0.3)
fig.savefig(FIGS_DIR / "fig_q1b_gap.png")
plt.close(fig)

# ---------------- Q2(a,b): CNN vs MLP clean/permuted ----------------
import json
q2a = json.load(open(RESULTS_DIR / "q2" / "q2a_summary.json"))
q2b = json.load(open(RESULTS_DIR / "q2" / "q2b_summary.json"))
fig, ax = plt.subplots(figsize=(3.4, 2.6))
names = ["q2-mlp", "q2-cnn"]
clean = [q2a[n]["test_acc"] for n in names]
perm = [q2b[n + "-permuted"]["test_acc"] for n in names]
x = np.arange(2)
ax.bar(x - 0.2, clean, 0.4, label="clean pixels", color=C_BLUE)
ax.bar(x + 0.2, perm, 0.4, label="permuted pixels", color=C_RED)
ax.set_xticks(x)
ax.set_xticklabels(["MLP", "CNN"])
ax.set_ylabel("test accuracy")
ax.set_ylim(0, 0.85)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.savefig(FIGS_DIR / "fig_q2_clean_vs_permuted.png")
plt.close(fig)

# ---------------- Q2(c): translation sweep ----------------
q2c = json.load(open(RESULTS_DIR / "q2" / "q2c_summary.json"))
ks = sorted(int(k) for k in q2c["q2-cnn"])
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3))
for name, color in [("q2-cnn", C_BLUE), ("q2-mlp", C_RED)]:
    accs = [q2c[name][str(k)]["acc"] for k in ks]
    axes[0].plot(ks, accs, marker="o", color=color, label=name)
    cc = [q2c[name][str(k)]["conf_correct"] for k in ks]
    cw = [q2c[name][str(k)]["conf_wrong"] for k in ks]
    axes[1].plot(ks, cc, marker="o", color=color, label=f"{name} conf|correct")
    axes[1].plot(ks, cw, marker="s", linestyle="--", color=color, label=f"{name} conf|wrong")
axes[0].set_xlabel("shift k (px)"); axes[0].set_ylabel("test accuracy"); axes[0].grid(alpha=0.3); axes[0].legend()
axes[1].set_xlabel("shift k (px)"); axes[1].set_ylabel("mean confidence"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=6)
ent = [q2c[name][str(k)]["flip_to_error"] for k in ks]
axes[2].plot(ks, [q2c["q2-cnn"][str(k)]["flip_to_error"] for k in ks], marker="o", color=C_BLUE, label="CNN")
axes[2].plot(ks, [q2c["q2-mlp"][str(k)]["flip_to_error"] for k in ks], marker="o", color=C_RED, label="MLP")
axes[2].set_xlabel("shift k (px)"); axes[2].set_ylabel("flip-to-error rate"); axes[2].grid(alpha=0.3)
axes[2].legend()
fig.savefig(FIGS_DIR / "fig_q2c_translation.png")
plt.close(fig)

# ---------------- Q3(a): val recon error vs bottleneck ----------------
q3a = json.load(open(RESULTS_DIR / "q3" / "q3a_summary.json"))
fig, ax = plt.subplots(figsize=(3.4, 2.4))
Bs = [q3a[k]["bottleneck"] for k in ["ae-b32", "ae-b256", "ae-b1024"]]
mse = [q3a[k]["val_mse"] for k in ["ae-b32", "ae-b256", "ae-b1024"]]
ax.plot(Bs, mse, marker="o", color=C_BLUE)
ax.set_xlabel("bottleneck size B")
ax.set_ylabel("validation MSE")
ax.set_xscale("log")
ax.grid(alpha=0.3)
fig.savefig(FIGS_DIR / "fig_q3a_valmse.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.2))
for i, k in enumerate(["ae-b32", "ae-b256", "ae-b1024"]):
    r = read_log(f"q3/{k}")
    ax.plot(col(r, "epoch", int), col(r, "val_mse"), label=k, linewidth=1)
ax.set_xlabel("epoch"); ax.set_ylabel("validation MSE"); ax.legend(); ax.grid(alpha=0.3)
fig.savefig(FIGS_DIR / "fig_q3a_curves.png")
plt.close(fig)

# ---------------- Q3(c): reconstruction vs class information ----------------
q3c = json.load(open(RESULTS_DIR / "q3" / "q3c_summary.json"))
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for k, v in q3c.items():
    marker = "o" if k.startswith("ae") else "s"
    ax.scatter(v["test_mse"], v["probe_test_acc_clean"], marker=marker, s=40)
    ax.annotate(k.replace("ae-", "AE-"), (v["test_mse"], v["probe_test_acc_clean"]),
                fontsize=6, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("test reconstruction MSE")
ax.set_ylabel("linear-probe test accuracy")
ax.grid(alpha=0.3)
fig.savefig(FIGS_DIR / "fig_q3c_probe_vs_mse.png")
plt.close(fig)

# ---------------- Q4(a): sweep plots ----------------
import csv as _csv
rows = list(_csv.DictReader(open(RESULTS_DIR / "q4" / "sweep.csv")))
kernels = ["linear", "rbf", "poly"]
colors = {"linear": C_GREEN, "rbf": C_BLUE, "poly": C_ORANGE}
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
for k in kernels:
    r = [x for x in rows if x["kernel"] == k]
    Cs = [float(x["C"]) for x in r]
    axes[0].plot(Cs, [float(x["val_acc"]) for x in r], marker="o", label=k, color=colors[k])
    axes[1].plot(Cs, [float(x["sv_fraction"]) for x in r], marker="o", label=k, color=colors[k])
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("C (log scale)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("validation accuracy")
axes[1].set_ylabel("support-vector fraction")
axes[0].legend(); axes[1].legend()
fig.savefig(FIGS_DIR / "fig_q4_sweep.png")
plt.close(fig)

# ---------------- Q4: SVM confusion matrix ----------------
q4b_data = json.load(open(RESULTS_DIR / "q4" / "q4b_summary.json"))
cm = np.array(q4b_data["confusion"])
fig, ax = plt.subplots(figsize=(3.4, 3.0))
from src.utils import CIFAR10_CLASSES
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels([c[:4] for c in CIFAR10_CLASSES], rotation=45, ha="right", fontsize=6)
ax.set_yticklabels([c[:4] for c in CIFAR10_CLASSES], fontsize=6)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.savefig(FIGS_DIR / "fig_q4_confusion.png")
plt.close(fig)

print("figures written to", FIGS_DIR)

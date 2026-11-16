"""Shared utilities: seeding, JSON IO, timing."""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

# Roll number P26AI0004 -> all numeric characters in order -> digits 2,6,0,0,0,4
RAW_ROLL = "P26AI0004"
SEED = int("".join(ch for ch in RAW_ROLL if ch.isdigit()))  # 260004

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
FIGS_DIR = REPO_ROOT / "figs"
SPLIT_DIR = REPO_ROOT / "split"
SPLIT_FILE = SPLIT_DIR / f"cifar10_split_seed{SEED}.npz"
PERMUTE_FILE = REPO_ROOT / "split" / f"pixel_permutation_seed{SEED}.npy"

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def load_json(path):
    with open(path) as f:
        return json.load(f)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0

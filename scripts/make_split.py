"""Create and verify the fixed stratified 45,000/5,000 split (seed = 260004).

Usage: python scripts/make_split.py
Output: split/cifar10_split_seed260004.npz  (+ printed class counts)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data import create_split, load_split_arrays
from src.utils import SEED


def main():
    tr, va = create_split()
    _, y, _, _ = load_split_arrays()
    print(f"seed (from roll number P26AI0004) = {SEED}")
    print(f"train size = {len(tr)}, val size = {len(va)}, overlap = {len(set(tr) & set(va))}")
    print("train class counts:", np.bincount(y[tr]))
    print("val   class counts:", np.bincount(y[va]))
    assert len(set(tr) | set(va)) == 50000
    assert len(tr) == 45000 and len(va) == 5000


if __name__ == "__main__":
    main()

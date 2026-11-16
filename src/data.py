"""CIFAR10 data loading, the fixed stratified 45k/5k split, and transforms.

Split rule (assignment): all numeric characters of the complete roll number
P26AI0004, in order, are used as the random seed -> SEED = 260004.
A stratified 45,000 train / 5,000 validation split is created from the 50,000
official training images. The official CIFAR10 test set is never used for
tuning. Split indices are saved to split/cifar10_split_seed260004.npz and
reused by every experiment.

Two image domains exist:
  * classification domain: ToTensor ([0,1]) followed by Normalize(mean, std)
  * autoencoder domain:    ToTensor only ([0,1])
No data augmentation is used anywhere, which keeps all comparisons fair.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms

from .utils import (CIFAR10_MEAN, CIFAR10_STD, DATA_DIR, PERMUTE_FILE,
                    SEED, SPLIT_FILE)

NORMALIZE = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
TO_TENSOR = transforms.ToTensor()


def get_raw_cifar10(train):
    return datasets.CIFAR10(root=str(DATA_DIR), train=train, download=True)


def create_split():
    """Create (or load) the fixed stratified 45k/5k split."""
    if SPLIT_FILE.exists():
        d = np.load(SPLIT_FILE)
        return d["train_idx"], d["val_idx"]

    from sklearn.model_selection import train_test_split

    full = get_raw_cifar10(train=True)
    targets = np.array(full.targets)
    idx = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        idx, test_size=5000, stratify=targets, random_state=SEED,
    )
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)

    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SPLIT_FILE, train_idx=train_idx, val_idx=val_idx,
             seed=SEED, roll_number="P26AI0004")
    return train_idx, val_idx


def get_pixel_permutation():
    """Fixed random permutation of the 3072 image coordinates (seed = SEED)."""
    if PERMUTE_FILE.exists():
        return np.load(PERMUTE_FILE)
    perm = np.random.RandomState(SEED).permutation(3 * 32 * 32)
    PERMUTE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(PERMUTE_FILE, perm)
    return perm


def load_split_arrays():
    """Return (Xtr, ytr, Xte, yte) as float32 arrays in [0,1] (NHWC) and int labels."""
    train = get_raw_cifar10(train=True)
    test = get_raw_cifar10(train=False)
    Xtr = train.data.astype(np.float32) / 255.0
    ytr = np.array(train.targets, dtype=np.int64)
    Xte = test.data.astype(np.float32) / 255.0
    yte = np.array(test.targets, dtype=np.int64)
    return Xtr, ytr, Xte, yte


def split_arrays():
    """Return arrays for the fixed split: Xtr/ytr (45k), Xval/yval (5k), Xte/yte (10k)."""
    Xtr, ytr, Xte, yte = load_split_arrays()
    tr_idx, va_idx = create_split()
    return Xtr[tr_idx], ytr[tr_idx], Xtr[va_idx], ytr[va_idx], Xte, yte


def normalize_batch(x):
    """x: (N,3,32,32) float tensor in [0,1] -> normalized tensor."""
    mean = torch.tensor(CIFAR10_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def add_gaussian_noise(x, sigma=0.25, seed=SEED):
    """Deterministic per-run Gaussian noise for [0,1] images (x: numpy or torch)."""
    if isinstance(x, np.ndarray):
        g = np.random.RandomState(seed)
        return np.clip(x + g.normal(0.0, sigma, x.shape).astype(np.float32), 0.0, 1.0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(x.shape, generator=g).to(x.device) * sigma
    return (x + noise).clamp(0.0, 1.0)


def shift_tensor(x, dx=0, dy=0):
    """Shift a normalized (N,3,H,W) batch by (dx, dy) pixels with zero padding."""
    n, c, h, w = x.shape
    pad_l, pad_r = max(dx, 0), max(-dx, 0)
    pad_t, pad_b = max(dy, 0), max(-dy, 0)
    xp = torch.nn.functional.pad(x, (pad_l, pad_r, pad_t, pad_b))
    ys = max(-dy, 0)
    xs = max(-dx, 0)
    return xp[:, :, ys:ys + h, xs:xs + w]


class _ArrayCIFAR(torch.utils.data.Dataset):
    """In-memory dataset for AE-domain training/eval. Images stored as
    (N,3,32,32) float32 tensors on CPU in [0,1] to keep memory small."""

    def __init__(self, images_nhwc):
        x = torch.from_numpy(np.ascontiguousarray(np.transpose(images_nhwc, (0, 3, 1, 2)))).float()
        self.x = x if x.is_contiguous() else x.contiguous()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i]


def ae_loaders(batch_size=128, num_workers=0, seed=SEED):
    """DataLoaders (train/val/test) yielding (x, x) pairs in [0,1] for autoencoders."""
    Xtr, ytr, Xval, yval, Xte, yte = split_arrays()

    def mk(arr):
        ds = _ArrayCIFAR(arr)
        return DataLoader(ds, batch_size=batch_size, shuffle=(arr is Xtr),
                          num_workers=num_workers,
                          generator=torch.Generator().manual_seed(seed) if arr is Xtr else None)

    return mk(Xtr), mk(Xval), mk(Xte)


class _TensorCache:
    data = {}


def get_tensor_split(normalize=True, permuted=False):
    """In-memory tensor views of the fixed split (fast path).

    Returns (x_train, y_train, x_val, y_val, x_test, y_test) where x are
    (N,3,32,32) float tensors in [0,1] (or normalized if normalize=True),
    y are (N,) int64 tensors. Cached; identical values to the loader path.
    """
    key = (normalize, permuted)
    if key in _TensorCache.data:
        return _TensorCache.data[key]
    Xtr, ytr, Xval, yval, Xte, yte = split_arrays()

    def to_t(a):
        return torch.from_numpy(np.ascontiguousarray(np.transpose(a, (0, 3, 1, 2)))).float()

    xtr, xval, xte = to_t(Xtr), to_t(Xval), to_t(Xte)
    if permuted:
        perm = torch.from_numpy(get_pixel_permutation())

        def permute(t):
            n = t.shape[0]
            return t.reshape(n, -1)[:, perm].reshape(n, 3, 32, 32).contiguous()

        xtr, xval, xte = permute(xtr), permute(xval), permute(xte)

    if normalize:
        mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
        xtr = (xtr - mean) / std
        xval = (xval - mean) / std
        xte = (xte - mean) / std
    out = (xtr, torch.from_numpy(ytr), xval, torch.from_numpy(yval),
           xte, torch.from_numpy(yte))
    _TensorCache.data[key] = out
    return out


def make_classifiers_loaders(permuted=False, batch_size=256, num_workers=0, seed=SEED):
    """Train/val/test loaders for the classification experiments (fast, in-memory)."""
    xtr, ytr, xval, yval, xte, yte = get_tensor_split(normalize=True, permuted=permuted)
    train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(TensorDataset(xval, yval), batch_size=1024, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorDataset(xte, yte), batch_size=1024, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader

"""EMNIST split and tensor helpers shared by vision training, data preparation, and NLP tuning."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from data.preprocess import fix_emnist_orientation

SPLIT_SEED = 42
TRAIN_FRAC = 0.9
_PIXEL_MAX = 255.0
_MIN_SPLIT_PER_CLASS = 2

__all__ = ["SPLIT_SEED", "TRAIN_FRAC", "compute_stratified_split", "extract_oriented_tensors"]


def _is_real(val: Any) -> bool:
    return not isinstance(val, bool) and isinstance(val, (int, float)) and math.isfinite(float(val))


def compute_stratified_split(targets: np.ndarray | torch.Tensor, train_frac: float = TRAIN_FRAC, seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Compute deterministic per-class stratified train/validation index arrays."""
    arr = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else targets
    if not isinstance(arr, np.ndarray):
        raise TypeError("targets must be a numpy.ndarray or torch.Tensor.")
    if arr.ndim != 1 or arr.size < _MIN_SPLIT_PER_CLASS or not np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        raise ValueError("targets must be a 1D integer array or tensor with at least 2 samples.")
    if not _is_real(train_frac) or not (0.0 < float(train_frac) < 1.0):
        raise ValueError("train_frac must be a finite number strictly between 0 and 1.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    rng, train_parts, val_parts = np.random.default_rng(seed), [], []
    for cls in np.unique(arr):
        cls_idx = np.flatnonzero(arr == cls)
        if cls_idx.size < _MIN_SPLIT_PER_CLASS:
            raise ValueError(f"Class {int(cls)} needs at least 2 samples for stratified split.")
        perm = rng.permutation(cls_idx)
        n_train = max(1, min(int(perm.size) - 1, round(float(perm.size) * float(train_frac))))
        train_parts.append(perm[:n_train])
        val_parts.append(perm[n_train:])
    return rng.permutation(np.concatenate(train_parts)).astype(np.int64, copy=False), rng.permutation(np.concatenate(val_parts)).astype(np.int64, copy=False)


def extract_oriented_tensors(emnist_ds: Any, indices: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw EMNIST dataset tensors into oriented (N, 1, 28, 28) float32 and int64 labels."""
    if not hasattr(emnist_ds, "data") or not hasattr(emnist_ds, "targets"):
        raise TypeError("emnist_ds must provide 'data' and 'targets' attributes.")
    raw_data, raw_targets = torch.as_tensor(emnist_ds.data, device="cpu"), torch.as_tensor(emnist_ds.targets, device="cpu")
    if raw_data.ndim != 3 or raw_data.shape[0] == 0 or raw_targets.ndim != 1 or raw_targets.shape[0] != raw_data.shape[0]:
        raise ValueError("emnist_ds.data (N, 28, 28) and emnist_ds.targets (N,) must be non-empty and aligned.")
    if indices is not None:
        if not isinstance(indices, (np.ndarray, torch.Tensor)):
            raise TypeError("indices must be a numpy.ndarray, torch.Tensor, or None.")
        is_np = isinstance(indices, np.ndarray) and np.issubdtype(indices.dtype, np.integer) and not np.issubdtype(indices.dtype, np.bool_)
        is_t = isinstance(indices, torch.Tensor) and not indices.is_floating_point() and not indices.is_complex() and indices.dtype != torch.bool
        idx_t = torch.from_numpy(indices.astype(np.int64, copy=False)) if isinstance(indices, np.ndarray) else indices.detach().to(device="cpu", dtype=torch.long)
        if idx_t.ndim != 1 or idx_t.numel() == 0 or not (is_np or is_t) or int(idx_t.min().item()) < 0 or int(idx_t.max().item()) >= int(raw_data.shape[0]):
            raise ValueError("indices must be a non-empty 1D integer index sequence within dataset bounds.")
        raw_data, raw_targets = raw_data[idx_t], raw_targets[idx_t]
    oriented = fix_emnist_orientation(raw_data.unsqueeze(1).to(torch.float32) / _PIXEL_MAX)
    return (oriented if isinstance(oriented, torch.Tensor) else torch.from_numpy(oriented)), raw_targets.detach().to(device="cpu", dtype=torch.long).clone()

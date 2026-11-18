"""Train, quantize, evaluate, and benchmark EMNIST character recognition models."""

from __future__ import annotations

import argparse
import copy
import errno
import json
import math
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import EMNIST

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data._io_errors import map_os_error as _map_os_error
from data.emnist_data import (
    SPLIT_SEED,
    TRAIN_FRAC,
    compute_stratified_split,
    extract_oriented_tensors,
)
from data.preprocess import (
    CANONICAL_36_CHARS,
    EMNIST_BALANCED_CHARS,
    fix_emnist_orientation,
    fold_47_to_36_probs,
)
from models.char_models import NUM_CLASSES, CharCNN, DenseMLP, save_fp32_checkpoint
from models.quantization import (
    benchmark_cpu_latency,
    get_model_disk_size_mb,
    get_quantization_engine,
    quantize_cnn_static,
    save_quantized_cnn,
)

EMNIST_ROOT, WEIGHTS_DIR, RESULTS_DIR = Path("data/emnist"), Path("weights"), Path("results")
SEED, BATCH_SIZE = SPLIT_SEED, 256
DEFAULT_MLP_EPOCHS, DEFAULT_CNN_EPOCHS, CPU_EVAL_THREADS = 20, 18, 4
AFFINE_DEGREES, AFFINE_TRANSLATE, AFFINE_SCALE = 10.0, (0.08, 0.08), (0.92, 1.08)

_MLP_LR, _MLP_MOMENTUM, _CNN_LR, _WEIGHT_DECAY = 0.05, 0.9, 1.5e-3, 1e-4
_BENCHMARK_WARMUP, _BENCHMARK_RUNS, _BENCHMARK_BATCH_SIZES = 10, 100, (1, 8)
_PCT_SCALE = 100.0
_DEG_TO_RAD, _DATASET_NAME = math.pi / 180.0, "emnist_balanced"
_CHAR36_TO_IDX = {ch: idx for idx, ch in enumerate(CANONICAL_36_CHARS)}
_FOLD_TARGET_MAP = torch.tensor([_CHAR36_TO_IDX[ch.upper()] for ch in EMNIST_BALANCED_CHARS], dtype=torch.long)

__all__ = [
    "BATCH_SIZE", "EMNIST_ROOT", "RESULTS_DIR", "SEED", "TRAIN_FRAC", "WEIGHTS_DIR",
    "apply_random_affine_batch", "build_emnist_transform", "compute_stratified_split",
    "evaluate_vision_model", "extract_oriented_tensors", "load_emnist_datasets",
    "main", "parse_args", "run_vision_training", "train_char_cnn", "train_dense_mlp",
]


def _is_real(val: Any) -> bool:
    return not isinstance(val, bool) and isinstance(val, (int, float)) and math.isfinite(float(val))


def _pos_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _path_arg(path: Any, name: str) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"{name} must be a string or os.PathLike object.")
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{name} must not be empty.")
    return Path(raw)


def build_emnist_transform(augment: bool = False) -> transforms.Compose:
    """Build EMNIST orientation transform with optional affine augmentation."""
    if not isinstance(augment, bool):
        raise TypeError("augment must be a boolean.")
    steps: list[Any] = [transforms.ToTensor(), transforms.Lambda(fix_emnist_orientation)]
    if augment:
        steps.append(transforms.RandomAffine(degrees=AFFINE_DEGREES, translate=AFFINE_TRANSLATE, scale=AFFINE_SCALE))
    return transforms.Compose(steps)


def load_emnist_datasets(root: str | os.PathLike[str] = EMNIST_ROOT) -> tuple[EMNIST, EMNIST, EMNIST]:
    """Load EMNIST Balanced train (augmented), val, and frozen test datasets."""
    root_path = _path_arg(root, "root")
    root_str = os.fspath(root_path)
    try:
        st = os.stat(root_path)
    except OSError as err:
        raise _map_os_error(err, root_str, "accessing EMNIST root") from err
    if not stat.S_ISDIR(st.st_mode):
        raise NotADirectoryError(errno.ENOTDIR, f"EMNIST root must be a directory: {root_str}")
    try:
        return (
            EMNIST(root=root_str, split="balanced", train=True, download=False, transform=build_emnist_transform(True)),
            EMNIST(root=root_str, split="balanced", train=True, download=False, transform=build_emnist_transform(False)),
            EMNIST(root=root_str, split="balanced", train=False, download=False, transform=build_emnist_transform(False)),
        )
    except OSError as err:
        raise _map_os_error(err, err.filename or root_str, "loading EMNIST dataset") from err
    except RuntimeError as exc:
        detail = " ".join(str(exc).splitlines()).strip()
        if "not found" in detail.lower():
            raise FileNotFoundError(errno.ENOENT, f"EMNIST balanced files not found under {root_str}.") from exc
        raise RuntimeError(f"Failed to load EMNIST balanced dataset from {root_str}: {detail or 'dataset error'}") from exc


def apply_random_affine_batch(
    batch: torch.Tensor, degrees: float = AFFINE_DEGREES, translate: tuple[float, float] = AFFINE_TRANSLATE,
    scale: tuple[float, float] = AFFINE_SCALE, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply independent per-sample random affine transforms via bilinear grid sampling."""
    if not isinstance(batch, torch.Tensor):
        raise TypeError("batch must be a torch.Tensor.")
    if batch.ndim != 4 or min(batch.shape[0], batch.shape[2], batch.shape[3]) < 1 or not batch.is_floating_point():
        raise ValueError("batch must be a floating-point tensor with shape (B, C, H, W).")
    if not _is_real(degrees) or float(degrees) < 0.0:
        raise ValueError("degrees must be a finite non-negative number.")
    if not isinstance(translate, (tuple, list)) or len(translate) != 2 or not all(_is_real(v) and 0.0 <= float(v) <= 1.0 for v in translate):
        raise ValueError("translate must be a (tx, ty) pair with finite fractions in [0.0, 1.0].")
    if not isinstance(scale, (tuple, list)) or len(scale) != 2 or not all(_is_real(v) for v in scale) or float(scale[0]) <= 0.0 or float(scale[0]) > float(scale[1]):
        raise ValueError("scale bounds must satisfy 0.0 < min_scale <= max_scale.")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None.")
    bsz, _, height, width = batch.shape
    dev, dtype = (generator.device if generator is not None else batch.device), batch.dtype
    rand = torch.rand((4, bsz), device=dev, dtype=dtype, generator=generator)
    angles, inv_s = ((rand[0] * 2.0 - 1.0) * float(degrees)) * _DEG_TO_RAD, 1.0 / (rand[1] * (float(scale[1]) - float(scale[0])) + float(scale[0]))
    tx, ty = (rand[2] * 2.0 - 1.0) * (2.0 * float(translate[0])), (rand[3] * 2.0 - 1.0) * (2.0 * float(translate[1]))
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    theta = torch.empty((bsz, 2, 3), device=dev, dtype=dtype)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = inv_s * cos_a, inv_s * (height / width) * sin_a, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = -inv_s * (width / height) * sin_a, inv_s * cos_a, ty
    grid = F.affine_grid(theta.to(device=batch.device) if theta.device != batch.device else theta, size=list(batch.shape), align_corners=False)
    return F.grid_sample(batch, grid, mode="bilinear", padding_mode="zeros", align_corners=False).clamp(0.0, 1.0)


def _evaluate_47_accuracy(model: nn.Module, x_dev: torch.Tensor, y_dev: torch.Tensor, batch_size: int = BATCH_SIZE) -> float:
    model.eval()
    correct, total = 0, int(x_dev.shape[0])
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            correct += int((model(x_dev[start : start + batch_size]).argmax(dim=-1) == y_dev[start : start + batch_size]).sum().item())
    return float(correct / total)


def _train_classifier(
    model: nn.Module, optimizer: Optimizer, epochs: int, train_x: torch.Tensor, train_y: torch.Tensor,
    val_x: torch.Tensor, val_y: torch.Tensor, device: torch.device, batch_size: int = BATCH_SIZE, augment: bool = False,
) -> tuple[nn.Module, float]:
    valid_epochs, valid_bs = _pos_int(epochs, "epochs"), _pos_int(batch_size, "batch_size")
    criterion, scheduler = nn.CrossEntropyLoss(), CosineAnnealingLR(optimizer, T_max=valid_epochs)
    tx, ty, vx, vy = train_x.to(device), train_y.to(device), val_x.to(device), val_y.to(device)
    n_train, best_val, best_state = int(tx.shape[0]), -1.0, copy.deepcopy(model.state_dict())
    for _ in range(valid_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, valid_bs):
            idx = perm[start : start + valid_bs]
            bx = apply_random_affine_batch(tx[idx]) if augment else tx[idx]
            optimizer.zero_grad(set_to_none=True)
            criterion(model(bx), ty[idx]).backward()
            optimizer.step()
        scheduler.step()
        val_acc = _evaluate_47_accuracy(model, vx, vy, batch_size=valid_bs)
        if val_acc > best_val:
            best_val, best_state = val_acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.cpu().eval(), best_val


def train_dense_mlp(
    train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor,
    device: torch.device, epochs: int = DEFAULT_MLP_EPOCHS, batch_size: int = BATCH_SIZE,
) -> tuple[DenseMLP, float]:
    """Train DenseMLP with SGD + CosineAnnealingLR and return CPU eval model and best val acc."""
    model = DenseMLP(num_classes=NUM_CLASSES).to(device)
    return _train_classifier(model, SGD(model.parameters(), lr=_MLP_LR, momentum=_MLP_MOMENTUM, weight_decay=_WEIGHT_DECAY), epochs, train_x, train_y, val_x, val_y, device, batch_size, False)  # type: ignore[return-value]


def train_char_cnn(
    train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor,
    device: torch.device, epochs: int = DEFAULT_CNN_EPOCHS, batch_size: int = BATCH_SIZE,
) -> tuple[CharCNN, float]:
    """Train CharCNN with AdamW + CosineAnnealingLR + affine augmentation and return CPU eval model."""
    model = CharCNN(num_classes=NUM_CLASSES).to(device)
    return _train_classifier(model, AdamW(model.parameters(), lr=_CNN_LR, weight_decay=_WEIGHT_DECAY), epochs, train_x, train_y, val_x, val_y, device, batch_size, True)  # type: ignore[return-value]


def evaluate_vision_model(model: nn.Module, test_x: torch.Tensor, test_y: torch.Tensor, batch_size: int = BATCH_SIZE) -> dict[str, float]:
    """Evaluate 47-class and folded 36-class Top-1 accuracies on CPU."""
    valid_bs = _pos_int(batch_size, "batch_size")
    if test_x.shape[0] != test_y.shape[0] or test_x.shape[0] == 0:
        raise ValueError("test_x and test_y must be non-empty with matching sample counts.")
    model.eval()
    total, targets_47 = int(test_x.shape[0]), test_y.detach().to(device="cpu", dtype=torch.long)
    targets_36, correct_47, correct_36 = _FOLD_TARGET_MAP[targets_47], 0, 0
    with torch.inference_mode():
        for start in range(0, total, valid_bs):
            probs_47 = torch.softmax(model(test_x[start : start + valid_bs].to(device="cpu", dtype=torch.float32)), dim=-1)
            probs_36 = fold_47_to_36_probs(probs_47)
            probs_36_t = probs_36 if isinstance(probs_36, torch.Tensor) else torch.as_tensor(probs_36)
            correct_47 += int((probs_47.argmax(dim=-1) == targets_47[start : start + valid_bs]).sum().item())
            correct_36 += int((probs_36_t.argmax(dim=-1) == targets_36[start : start + valid_bs]).sum().item())
    acc_47, acc_36 = float(correct_47 / total), float(correct_36 / total)
    return {"test_acc_47": acc_47, "test_acc_47_pct": acc_47 * _PCT_SCALE, "test_acc_36": acc_36, "test_acc_36_pct": acc_36 * _PCT_SCALE}


def _make_latency_batches(inputs: torch.Tensor, batch_size: int, count: int) -> list[torch.Tensor]:
    total = int(inputs.shape[0])
    if total < 1:
        raise ValueError("inputs must contain at least one sample for benchmarking.")
    cpu_in = inputs.detach().to(device="cpu", dtype=torch.float32)
    if total < batch_size:
        cpu_in = cpu_in.repeat(((batch_size + total - 1) // total, 1, 1, 1))
        total = int(cpu_in.shape[0])
    batches: list[torch.Tensor] = []
    for idx in range(count):
        s = (idx * batch_size) % total
        batches.append(cpu_in[s : s + batch_size] if s + batch_size <= total else torch.cat([cpu_in[s:], cpu_in[: s + batch_size - total]], dim=0))
    return batches


def _benchmark_model(
    model: nn.Module, ckpt_path: Path, test_x: torch.Tensor, test_y: torch.Tensor, precision: str, best_val_acc: float | None = None
) -> dict[str, Any]:
    entry: dict[str, Any] = {"checkpoint": ckpt_path.as_posix(), "precision": precision, "disk_size_mb": float(get_model_disk_size_mb(ckpt_path)), **evaluate_vision_model(model, test_x, test_y, BATCH_SIZE)}
    if best_val_acc is not None:
        entry["best_val_acc_47"], entry["best_val_acc_47_pct"] = float(best_val_acc), float(best_val_acc) * _PCT_SCALE
    for bsz in _BENCHMARK_BATCH_SIZES:
        entry[f"latency_cpu_batch_{bsz}"] = benchmark_cpu_latency(model, _make_latency_batches(test_x, bsz, _BENCHMARK_WARMUP + _BENCHMARK_RUNS), warmup=_BENCHMARK_WARMUP, runs=_BENCHMARK_RUNS)
    return entry


def _finalize_metrics(
    weights_path: Path, results_path: Path, mlp_fp32: DenseMLP, mlp_val: float, cnn_fp32: CharCNN, cnn_val: float,
    cnn_int8: nn.Module, test_x: torch.Tensor, test_y: torch.Tensor, split_counts: dict[str, int],
    engine: str, device_type: str, epochs_mlp: int, epochs_cnn: int,
) -> dict[str, Any]:
    torch.set_num_threads(CPU_EVAL_THREADS)
    mlp_ckpt = save_fp32_checkpoint(mlp_fp32, weights_path / "dense_mlp.pt", metadata={"seed": SEED, "epochs": epochs_mlp})
    cnn_ckpt = save_fp32_checkpoint(cnn_fp32, weights_path / "char_cnn_fp32.pt", metadata={"seed": SEED, "epochs": epochs_cnn})
    int8_ckpt = save_quantized_cnn(cnn_int8, weights_path / "char_cnn_int8.pt")
    models = {
        "dense_mlp_fp32": _benchmark_model(mlp_fp32, mlp_ckpt, test_x, test_y, "fp32", mlp_val),
        "char_cnn_fp32": _benchmark_model(cnn_fp32, cnn_ckpt, test_x, test_y, "fp32", cnn_val),
        "char_cnn_int8": _benchmark_model(cnn_int8, int8_ckpt, test_x, test_y, "int8"),
    }
    mlp_m, cnn_m, int8_m = models["dense_mlp_fp32"], models["char_cnn_fp32"], models["char_cnn_int8"]
    if float(int8_m["disk_size_mb"]) <= 0.0:
        raise ValueError("INT8 checkpoint size must be positive.")
    metrics = {
        "dataset": _DATASET_NAME, "seed": SEED, "split_counts": split_counts, "quantization_engine": engine,
        "cpu_threads": CPU_EVAL_THREADS, "training_device": device_type, "models": models,
        "comparisons": {
            "cnn_vs_mlp_acc_47_gain_pp": float(cnn_m["test_acc_47_pct"] - mlp_m["test_acc_47_pct"]),
            "int8_cnn_size_reduction_x": float(cnn_m["disk_size_mb"] / float(int8_m["disk_size_mb"])),
            "int8_cnn_acc_47_drop_pp": float(cnn_m["test_acc_47_pct"] - int8_m["test_acc_47_pct"]),
            "int8_cnn_acc_36_drop_pp": float(cnn_m["test_acc_36_pct"] - int8_m["test_acc_36_pct"]),
        },
    }
    metrics_path = results_path / "vision_metrics.json"
    try:
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
            handle.write("\n")
    except OSError as err:
        raise _map_os_error(err, str(metrics_path), "writing vision metrics") from err
    return metrics


def run_vision_training(
    emnist_root: str | os.PathLike[str] = EMNIST_ROOT, weights_dir: str | os.PathLike[str] = WEIGHTS_DIR,
    results_dir: str | os.PathLike[str] = RESULTS_DIR, epochs_mlp: int = DEFAULT_MLP_EPOCHS, epochs_cnn: int = DEFAULT_CNN_EPOCHS,
) -> dict[str, Any]:
    """Run full vision training, static INT8 quantization, CPU benchmarking, and artifact export."""
    valid_mlp, valid_cnn = _pos_int(epochs_mlp, "epochs_mlp"), _pos_int(epochs_cnn, "epochs_cnn")
    weights_path, results_path = _path_arg(weights_dir, "weights_dir"), _path_arg(results_dir, "results_dir")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    _, val_ds, test_ds = load_emnist_datasets(emnist_root)
    for label, dpath in (("weights_dir", weights_path), ("results_dir", results_path)):
        try:
            dpath.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            raise _map_os_error(err, str(dpath), f"creating {label}") from err
    for target in (weights_path / "dense_mlp.pt", weights_path / "char_cnn_fp32.pt", weights_path / "char_cnn_int8.pt", results_path / "vision_metrics.json"):
        if target.is_dir():
            raise IsADirectoryError(errno.EISDIR, f"Expected a file path, got a directory: {target}")
    train_idx, val_idx = compute_stratified_split(val_ds.targets, train_frac=TRAIN_FRAC, seed=SEED)
    train_x, train_y = extract_oriented_tensors(val_ds, train_idx)
    val_x, val_y = extract_oriented_tensors(val_ds, val_idx)
    test_x, test_y = extract_oriented_tensors(test_ds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_fp32, mlp_val = train_dense_mlp(train_x, train_y, val_x, val_y, device, epochs=valid_mlp)
    cnn_fp32, cnn_val = train_char_cnn(train_x, train_y, val_x, val_y, device, epochs=valid_cnn)
    engine = get_quantization_engine()
    cnn_int8 = quantize_cnn_static(cnn_fp32, DataLoader(TensorDataset(val_x), batch_size=BATCH_SIZE, shuffle=False), engine=engine)
    return _finalize_metrics(weights_path, results_path, mlp_fp32, mlp_val, cnn_fp32, cnn_val, cnn_int8, test_x, test_y, {"train": int(train_x.shape[0]), "val": int(val_x.shape[0]), "test": int(test_x.shape[0])}, engine, device.type, valid_mlp, valid_cnn)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for EMNIST vision model training and quantization."""
    p = argparse.ArgumentParser(description="Train DenseMLP, CharCNN, and static INT8 CharCNN.")
    for flag, typ, default, desc in (
        ("--epochs-mlp", int, DEFAULT_MLP_EPOCHS, "DenseMLP training epochs."),
        ("--epochs-cnn", int, DEFAULT_CNN_EPOCHS, "CharCNN training epochs."),
        ("--weights-dir", Path, WEIGHTS_DIR, "Checkpoint output directory."),
        ("--results-dir", Path, RESULTS_DIR, "Metrics JSON output directory."),
        ("--emnist-root", Path, EMNIST_ROOT, "EMNIST dataset root directory."),
    ):
        p.add_argument(flag, type=typ, default=default, help=desc)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for vision model training, quantization, and evaluation."""
    try:
        args = parse_args(argv)
        metrics = run_vision_training(args.emnist_root, args.weights_dir, args.results_dir, args.epochs_mlp, args.epochs_cnn)
    except OSError as exc:
        msg = " ".join((exc.strerror or str(exc)).splitlines()).strip()
        msg = f"{msg} ({exc.filename})" if exc.filename and exc.filename not in msg else msg
        sys.stderr.write(f"File system error: {msg or 'Filesystem operation failed.'}\n")
        return 1
    except (ValueError, TypeError, RuntimeError) as exc:
        sys.stderr.write(f"Error: {' '.join(str(exc).splitlines()).strip() or 'Vision training failed.'}\n")
        return 1
    m, c = metrics["models"], metrics["comparisons"]
    mlp, cnn, i8 = m["dense_mlp_fp32"], m["char_cnn_fp32"], m["char_cnn_int8"]
    sys.stdout.write(
        f"Vision complete ({metrics['training_device']}/{metrics['quantization_engine']}) | "
        f"MLP: {mlp['test_acc_47_pct']:.2f}%/{mlp['test_acc_36_pct']:.2f}% ({mlp['disk_size_mb']:.2f}MB) | "
        f"CNN: {cnn['test_acc_47_pct']:.2f}%/{cnn['test_acc_36_pct']:.2f}% ({cnn['disk_size_mb']:.2f}MB) | "
        f"INT8: {i8['test_acc_47_pct']:.2f}%/{i8['test_acc_36_pct']:.2f}% ({i8['disk_size_mb']:.2f}MB) | "
        f"{c['int8_cnn_size_reduction_x']:.2f}x smaller, {c['int8_cnn_acc_47_drop_pp']:.2f}pp drop\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

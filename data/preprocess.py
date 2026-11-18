"""Shared image validation, worksheet strip slicing, and 28x28 preprocessing."""

from __future__ import annotations

import errno
import io
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from data._box_ops import (
    BLANK_INTENSITY_THRESH,
    BLANK_MIN_ACTIVE_PIXELS,
    BLANK_MIN_INK_MASS,
    compute_border_trims,
    enhance_thin_strokes,
    normalize_box_polarity,
    resize_and_center_28x28,
    tight_bbox,
    to_float_grayscale_255,
)
from data._strip_ops import (
    crop_box_clean,
    detect_strip_grid,
    detect_whitespace_runs,
    select_real_boxes,
    strip_input_to_pil,
    strip_polarity_map,
    trim_outer_frame_x,
)

EMNIST_BALANCED_CHARS: tuple[str, ...] = tuple("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")
CANONICAL_36_CHARS: tuple[str, ...] = tuple("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MAX_IMAGE_BYTES: int = 5 * 1024 * 1024
MAX_IMAGE_DIM: int = 4096

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_NUM_EMNIST = len(EMNIST_BALANCED_CHARS)
_NUM_CANON = len(CANONICAL_36_CHARS)
_CHAR_TO_36 = {ch: idx for idx, ch in enumerate(CANONICAL_36_CHARS)}
_FOLD_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (i, _CHAR_TO_36[ch.upper()]) for i, ch in enumerate(EMNIST_BALANCED_CHARS) if i >= _NUM_CANON
)
_CANVAS_SIZE = 28
_STRIP_INK_ACTIVE_THRESH = 0.20


def _validate_image_limits(max_bytes: int, max_dim: int) -> None:
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer.")
    if not isinstance(max_dim, int) or isinstance(max_dim, bool) or max_dim <= 0:
        raise ValueError("max_dim must be a positive integer.")


def _fold_47_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim not in (1, 2) or tensor.shape[-1] != _NUM_EMNIST:
        raise ValueError("probs_47 must have shape (47,) or (N, 47).")
    if tensor.is_complex() or tensor.dtype == torch.bool:
        raise ValueError("probs_47 must contain real numeric values.")
    out_dtype = tensor.dtype if tensor.is_floating_point() else torch.float32
    work = tensor.detach().to(dtype=torch.float64 if out_dtype == torch.float64 else torch.float32)
    if not bool(torch.isfinite(work).all()) or bool((work < 0).any()):
        raise ValueError("probs_47 must contain finite non-negative values.")
    folded = work[..., :_NUM_CANON].clone()
    for low_i, up_i in _FOLD_PAIRS:
        folded[..., up_i] = folded[..., up_i] + work[..., low_i]
    sums = folded.sum(dim=-1, keepdim=True)
    zero_mask = sums <= 0
    normalized = folded / torch.where(zero_mask, torch.ones_like(sums), sums)
    uniform = torch.full_like(normalized, 1.0 / _NUM_CANON)
    return torch.where(zero_mask.expand_as(normalized), uniform, normalized).to(dtype=out_dtype)


def _fold_47_array(probs_47: Any) -> np.ndarray:
    if isinstance(probs_47, (str, bytes, bytearray, dict)):
        raise ValueError("probs_47 must be a numeric array, tensor, or sequence.")
    try:
        arr = np.asarray(probs_47)
    except (TypeError, ValueError) as exc:
        raise ValueError("probs_47 could not be converted to a numeric array.") from exc
    if arr.ndim not in (1, 2) or arr.shape[-1] != _NUM_EMNIST:
        raise ValueError("probs_47 must have shape (47,) or (N, 47).")
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError("probs_47 must contain real numeric values.")
    out_dtype = np.float64 if arr.dtype == np.float64 else np.float32
    work = arr.astype(out_dtype, copy=False)
    if not np.all(np.isfinite(work)) or np.any(work < 0):
        raise ValueError("probs_47 must contain finite non-negative values.")
    folded = np.array(work[..., :_NUM_CANON], dtype=out_dtype, copy=True)
    for low_i, up_i in _FOLD_PAIRS:
        folded[..., up_i] += work[..., low_i]
    sums = folded.sum(axis=-1, keepdims=True)
    zero_mask = sums <= 0
    normalized = folded / np.where(zero_mask, np.ones_like(sums), sums)
    uniform = np.full_like(normalized, 1.0 / _NUM_CANON, dtype=out_dtype)
    return np.where(np.broadcast_to(zero_mask, normalized.shape), uniform, normalized).astype(out_dtype)


def fold_47_to_36_probs(probs_47: Any) -> torch.Tensor | np.ndarray:
    """Fold 47-class EMNIST probabilities onto canonical 36 classes and re-normalize."""
    if isinstance(probs_47, torch.Tensor):
        return _fold_47_tensor(probs_47)
    return _fold_47_array(probs_47)


def fix_emnist_orientation(tensor_or_arr: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Apply .transpose(-2, -1) to raw EMNIST tensors or arrays without mutating input."""
    if not isinstance(tensor_or_arr, (torch.Tensor, np.ndarray)):
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray.")
    if tensor_or_arr.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions to transpose spatial axes.")
    if isinstance(tensor_or_arr, torch.Tensor):
        return tensor_or_arr.transpose(-2, -1).contiguous().clone()
    return np.swapaxes(tensor_or_arr, -2, -1).copy()


def load_validated_image_bytes(
    raw_bytes: bytes,
    max_bytes: int = MAX_IMAGE_BYTES,
    max_dim: int = MAX_IMAGE_DIM,
) -> Image.Image:
    """Validate raw PNG/JPEG bytes and header dimensions before decoding into a PIL Image."""
    _validate_image_limits(max_bytes, max_dim)
    if not isinstance(raw_bytes, (bytes, bytearray)):
        raise ValueError("Image data must be bytes or bytearray.")
    if len(raw_bytes) == 0 or len(raw_bytes) > max_bytes:
        raise ValueError(f"Image size must be between 1 and {max_bytes} bytes.")
    payload = bytes(raw_bytes)
    if not payload.startswith((_PNG_MAGIC, _JPEG_MAGIC)):
        raise ValueError("Unsupported image format: only PNG and JPEG are allowed.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(io.BytesIO(payload)) as img:
                if img.format not in ("PNG", "JPEG"):
                    raise ValueError("Unsupported image format: only PNG and JPEG are allowed.")
                w, h = img.size
                if w <= 0 or h <= 0 or w > max_dim or h > max_dim:
                    raise ValueError(f"Image dimensions {w}x{h} exceed maximum {max_dim}x{max_dim}.")
                img.load()
                transposed = ImageOps.exif_transpose(img)
                result = transposed.copy()
                if transposed is not img:
                    transposed.close()
                return result
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            UnidentifiedImageError,
            OSError,
            SyntaxError,
            EOFError,
        ) as exc:
            raise ValueError("Invalid, corrupted, or oversized PNG/JPEG image data.") from exc


def load_validated_image_file(
    path: str | os.PathLike[str],
    max_bytes: int = MAX_IMAGE_BYTES,
    max_dim: int = MAX_IMAGE_DIM,
) -> Image.Image:
    """Read and validate a PNG/JPEG file from disk with explicit OS error mapping."""
    _validate_image_limits(max_bytes, max_dim)
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("Image path must be a string or path-like object.")
    path_label = os.fspath(path)
    try:
        with open(path, "rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError as err:
        if err.errno == errno.ENOENT:
            raise FileNotFoundError(errno.ENOENT, f"Image file not found: {path_label}") from err
        if err.errno == errno.EISDIR or (err.errno == errno.EACCES and Path(path).is_dir()):
            raise IsADirectoryError(
                errno.EISDIR, f"Expected an image file, got a directory: {path_label}"
            ) from err
        if err.errno == errno.EACCES:
            raise PermissionError(
                errno.EACCES, f"Permission denied reading image file: {path_label}"
            ) from err
        raise OSError(err.errno, f"Unable to read image file: {path_label}") from err
    return load_validated_image_bytes(raw, max_bytes=max_bytes, max_dim=max_dim)


def preprocess_box(img_input: Any) -> tuple[torch.Tensor, bool]:
    """Preprocess one character box into a (1, 28, 28) EMNIST-ready tensor and empty flag."""
    gray = to_float_grayscale_255(img_input)
    top, bot, left, right = compute_border_trims(gray)
    height, width = gray.shape
    norm = normalize_box_polarity(gray[top : height - bot, left : width - right])
    active = norm > BLANK_INTENSITY_THRESH
    count = int(active.sum())
    mass = float(norm[active].sum()) if count > 0 else 0.0
    if float(norm.max()) < BLANK_INTENSITY_THRESH or count < BLANK_MIN_ACTIVE_PIXELS or mass < BLANK_MIN_INK_MASS:
        return torch.zeros((1, _CANVAS_SIZE, _CANVAS_SIZE), dtype=torch.float32), True
    canvas = resize_and_center_28x28(enhance_thin_strokes(tight_bbox(norm)))
    return torch.from_numpy(canvas).unsqueeze(0), False


def slice_strip_image(
    strip_img: Image.Image | np.ndarray,
    num_boxes: int | None = None,
) -> list[Image.Image]:
    """Slice a horizontal worksheet strip into ordered character-box PIL Image crops."""
    if num_boxes is not None and (
        not isinstance(num_boxes, int) or isinstance(num_boxes, bool) or num_boxes < 1
    ):
        raise ValueError("num_boxes must be None or a positive integer.")
    clean_pil = strip_input_to_pil(strip_img)
    gray = np.asarray(clean_pil if clean_pil.mode == "L" else clean_pil.convert("L"), dtype=np.float32)
    norm = strip_polarity_map(gray)
    active = norm > _STRIP_INK_ACTIVE_THRESH
    y_start, y_end, top_rows, bot_rows, div_runs = detect_strip_grid(active)
    grid_boxes = select_real_boxes(active, norm, y_start, y_end, top_rows, bot_rows, div_runs)
    if num_boxes is not None:
        if len(grid_boxes) == num_boxes:
            intervals = grid_boxes
        else:
            x_start, x_end = trim_outer_frame_x(active, y_start, y_end, div_runs, grid_boxes)
            span = x_end - x_start
            intervals = [
                (
                    min(x_end - 1, x_start + round(i * span / num_boxes)),
                    min(x_end, max(x_start + round(i * span / num_boxes) + 1, x_start + round((i + 1) * span / num_boxes))),
                )
                for i in range(num_boxes)
            ]
    elif grid_boxes and (top_rows or bot_rows or len(grid_boxes) >= 2):
        intervals = grid_boxes
    else:
        x_start, x_end = trim_outer_frame_x(active, y_start, y_end, div_runs, grid_boxes)
        runs = detect_whitespace_runs(norm[y_start:y_end, x_start:x_end], x_start)
        intervals = runs if runs else [(x_start, x_end)]
    return [
        crop_box_clean(clean_pil, gray, y_start, y_end, x0, x1)
        for x0, x1 in intervals
    ]

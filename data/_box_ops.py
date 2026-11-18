"""Internal image conversion, border-stripping, and 28x28 glyph formatting helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

MAX_BYTE_VAL = 255.0
POLARITY_MIDPOINT = 127.0
MIN_BORDER_STRIP_SIDE = 8
BLANK_INTENSITY_THRESH = 0.12
BLANK_MIN_ACTIVE_PIXELS = 3
BLANK_MIN_INK_MASS = 1.5

_CONTRAST_DIFF_THRESH = 35.0
_OUTER_BORDER_FRAC = 0.12
_TILTED_BAND_FRAC = 0.10
_MAX_BORDER_TRIM_FRAC = 0.15
_ROW_BORDER_OCC_THRESH = 0.65
_BAND_UNION_OCC_THRESH = 0.70
_BAND_ACTIVE_OCC_THRESH = 0.03
_TIGHT_FALLBACK_FRAC = 0.05
_FG_STROKE_THRESH = 0.20
_FG_FALLBACK_THRESH = 0.10
_THIN_STROKE_PROJ_HALF_W = 1.1
_VERY_THIN_STROKE_PROJ_HALF_W = 0.45
_THIN_STROKE_MIN_DIM = 20
_DILATE_PAD_PX = 4
_GAUSSIAN_BLUR_SIGMA = 0.6
_TARGET_GLYPH_BOX = 20
_CANVAS_SIZE = 28
_CANVAS_COM_CENTER = 13.5


def find_contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    """Group sorted 1D integer indices into half-open contiguous (start, end) runs."""
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = start
    for raw_idx in indices[1:]:
        idx = int(raw_idx)
        if idx == prev + 1:
            prev = idx
        else:
            runs.append((start, prev + 1))
            start = prev = idx
    runs.append((start, prev + 1))
    return runs


def pil_to_detached_clean(img: Image.Image, prefer_rgb: bool = False) -> Image.Image:
    """Composite any alpha/transparency onto white and return a copied RGB or L Image."""
    if img.width <= 0 or img.height <= 0:
        raise ValueError("Image dimensions must be positive.")
    has_palette_alpha = img.mode == "P" and (
        "transparency" in img.info or bool(getattr(img, "has_transparency_data", False))
    )
    if img.mode in ("RGBA", "LA", "PA", "RGBa", "La") or has_palette_alpha:
        rgba = img.convert("RGBA")
        white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        target_mode = "L" if (img.mode in ("LA", "La") and not prefer_rgb) else "RGB"
        composited = Image.alpha_composite(white_bg, rgba).convert(target_mode)
        rgba.close()
        white_bg.close()
        return composited
    if img.mode in ("L", "RGB"):
        return img.copy()
    return img.convert("RGB" if prefer_rgb else "L")


def normalize_array_to_hwc(arr: np.ndarray, is_tensor: bool) -> np.ndarray:
    """Validate an image array/tensor and return a float32 [0..255] 2D or HWC copy."""
    if arr.size == 0 or arr.ndim not in (2, 3):
        raise ValueError("Image array must be non-empty 2D or 3D.")
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError("Image array must have a real numeric dtype.")
    work = arr.astype(np.float32, copy=True)
    if not np.all(np.isfinite(work)):
        raise ValueError("Image array contains non-finite values.")
    if np.issubdtype(arr.dtype, np.floating) and float(work.min()) >= 0.0 and float(work.max()) <= 1.0:
        work *= MAX_BYTE_VAL
    work = np.clip(work, 0.0, MAX_BYTE_VAL)
    if work.ndim == 2:
        return work
    c_last = work.shape[2] in (1, 2, 3, 4)
    c_first = work.shape[0] in (1, 2, 3, 4)
    if c_last and (not c_first or (not is_tensor and work.shape[0] > 4)):
        return work
    if c_first and not c_last:
        return np.transpose(work, (1, 2, 0))
    if c_first and c_last:
        return np.transpose(work, (1, 2, 0)) if is_tensor else work
    raise ValueError("3D image array must have 1, 2, 3, or 4 channels.")


def hwc_to_float_gray(hwc: np.ndarray) -> np.ndarray:
    """Composite alpha onto white and convert a 2D/HWC float32 [0..255] array to grayscale."""
    if hwc.ndim == 2:
        return hwc.astype(np.float32, copy=False)
    channels = hwc.shape[2]
    if channels == 1:
        return hwc[:, :, 0].astype(np.float32, copy=False)
    if channels == 2:
        alpha = np.clip(hwc[:, :, 1] / MAX_BYTE_VAL, 0.0, 1.0)
        return (hwc[:, :, 0] * alpha + MAX_BYTE_VAL * (1.0 - alpha)).astype(np.float32)
    lum = 0.299 * hwc[:, :, 0] + 0.587 * hwc[:, :, 1] + 0.114 * hwc[:, :, 2]
    if channels == 4:
        alpha = np.clip(hwc[:, :, 3] / MAX_BYTE_VAL, 0.0, 1.0)
        return (lum * alpha + MAX_BYTE_VAL * (1.0 - alpha)).astype(np.float32)
    return lum.astype(np.float32)


def to_float_grayscale_255(img_input: Any) -> np.ndarray:
    """Convert PIL Image, NumPy array, or PyTorch tensor to float32 grayscale (H, W) [0..255]."""
    if isinstance(img_input, Image.Image):
        clean = pil_to_detached_clean(img_input, prefer_rgb=False)
        gray_pil = clean if clean.mode == "L" else clean.convert("L")
        return np.asarray(gray_pil, dtype=np.float32).copy()
    if isinstance(img_input, torch.Tensor):
        hwc = normalize_array_to_hwc(img_input.detach().cpu().numpy(), is_tensor=True)
        return hwc_to_float_gray(hwc)
    if isinstance(img_input, np.ndarray):
        hwc = normalize_array_to_hwc(img_input, is_tensor=False)
        return hwc_to_float_gray(hwc)
    raise ValueError("Input must be a PIL.Image.Image, numpy.ndarray, or torch.Tensor.")


def compute_side_trim(side_mask: np.ndarray) -> int:
    """Find border trim depth along axis 0 of a boolean contrast mask, capped at 15%."""
    side_len = side_mask.shape[0]
    if side_len < MIN_BORDER_STRIP_SIDE:
        return 0
    max_trim = int(_MAX_BORDER_TRIM_FRAC * side_len)
    outer_12 = min(max_trim, max(1, round(_OUTER_BORDER_FRAC * side_len)))
    row_occ = side_mask[:outer_12].mean(axis=1)
    high_rows = np.where(row_occ >= _ROW_BORDER_OCC_THRESH)[0]
    if high_rows.size > 0:
        return min(max_trim, int(high_rows[-1]) + 1)
    outer_10 = min(max_trim, max(1, int(np.ceil(_TILTED_BAND_FRAC * side_len))))
    active_rows = side_mask[:outer_10].mean(axis=1) >= _BAND_ACTIVE_OCC_THRESH
    for start, end in find_contiguous_runs(np.where(active_rows)[0]):
        if float(np.any(side_mask[start:end], axis=0).mean()) >= _BAND_UNION_OCC_THRESH:
            return min(max_trim, end)
    return 0


def compute_border_trims(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Compute (top, bottom, left, right) border trim depths relative to interior median."""
    height, width = gray.shape
    y0, y1 = max(1, round(0.15 * height)), min(height - 1, round(0.85 * height))
    x0, x1 = max(1, round(0.15 * width)), min(width - 1, round(0.85 * width))
    interior = gray[y0:y1, x0:x1] if (y1 > y0 and x1 > x0) else gray
    interior_med = float(np.median(interior))
    contrasting = np.abs(gray - interior_med) > _CONTRAST_DIFF_THRESH
    top = compute_side_trim(contrasting)
    bottom = compute_side_trim(contrasting[::-1, :])
    left = compute_side_trim(contrasting.T)
    right = compute_side_trim(contrasting.T[::-1, :])
    if top + bottom >= height:
        top, bottom = 0, 0
    if left + right >= width:
        left, right = 0, 0
    return top, bottom, left, right


def outer_ring_values(arr: np.ndarray) -> np.ndarray:
    """Return the 1-pixel perimeter values of a 2D array."""
    height, width = arr.shape
    if height < 2 or width < 2:
        return arr.ravel()
    return np.concatenate((arr[0, :], arr[-1, :], arr[1:-1, 0], arr[1:-1, -1]))


def normalize_box_polarity(crop: np.ndarray) -> np.ndarray:
    """Invert bright backgrounds and scale background-subtracted ink into [0.0, 1.0]."""
    ring_med = float(np.median(outer_ring_values(crop)))
    oriented = (MAX_BYTE_VAL - crop) if ring_med > POLARITY_MIDPOINT else crop
    bg = float(np.median(outer_ring_values(oriented)))
    denom = max(MAX_BYTE_VAL - bg, 1.0)
    return np.clip((oriented - bg) / denom, 0.0, 1.0).astype(np.float32)


def tight_bbox(arr: np.ndarray) -> np.ndarray:
    """Crop tightly around ink above threshold with a peak-relative fallback."""
    mask = arr > BLANK_INTENSITY_THRESH
    if not np.any(mask):
        mask = arr > (_TIGHT_FALLBACK_FRAC * float(arr.max()))
    ys, xs = np.where(mask)
    return arr[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]


def enhance_thin_strokes(cropped: np.ndarray) -> np.ndarray:
    """Lightly dilate and smooth thin high-resolution strokes to match EMNIST thickness."""
    bh, bw = cropped.shape
    if max(bh, bw) <= _THIN_STROKE_MIN_DIM:
        return cropped
    fg = cropped > _FG_STROKE_THRESH
    if not np.any(fg):
        fg = cropped > _FG_FALLBACK_THRESH
    if not np.any(fg):
        return cropped
    fg_padded = np.pad(fg, 1, mode="constant", constant_values=False)
    edt = ndimage.distance_transform_edt(fg_padded)[1:-1, 1:-1]
    projected_half_w = float(np.median(edt[fg])) * (float(_TARGET_GLYPH_BOX) / float(max(bh, bw)))
    if projected_half_w >= _THIN_STROKE_PROJ_HALF_W:
        return cropped
    kernel_size = 5 if projected_half_w < _VERY_THIN_STROKE_PROJ_HALF_W else 3
    padded = np.pad(cropped, _DILATE_PAD_PX, mode="constant", constant_values=0.0)
    dilated = ndimage.grey_dilation(padded, size=(kernel_size, kernel_size))
    smoothed = ndimage.gaussian_filter(dilated, sigma=_GAUSSIAN_BLUR_SIGMA)
    return tight_bbox(smoothed)


def resize_and_center_28x28(cropped: np.ndarray) -> np.ndarray:
    """Resize preserving aspect into 20x20 and center by center-of-mass on 28x28."""
    bh, bw = cropped.shape
    scale = float(_TARGET_GLYPH_BOX) / float(max(bh, bw))
    new_h = max(1, min(_TARGET_GLYPH_BOX, round(bh * scale)))
    new_w = max(1, min(_TARGET_GLYPH_BOX, round(bw * scale)))
    pil_crop = Image.fromarray(cropped.astype(np.float32, copy=False))
    resized_pil = pil_crop.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized = np.clip(np.asarray(resized_pil, dtype=np.float32), 0.0, 1.0).copy()
    peak = float(resized.max())
    if peak > 0.0:
        resized /= peak
    cy, cx = ndimage.center_of_mass(resized)
    if not np.isfinite(cy) or not np.isfinite(cx):
        cy, cx = (new_h - 1) / 2.0, (new_w - 1) / 2.0
    top = max(0, min(_CANVAS_SIZE - new_h, round(_CANVAS_COM_CENTER - float(cy))))
    left = max(0, min(_CANVAS_SIZE - new_w, round(_CANVAS_COM_CENTER - float(cx))))
    canvas = np.zeros((_CANVAS_SIZE, _CANVAS_SIZE), dtype=np.float32)
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas

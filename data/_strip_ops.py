"""Internal helpers for slicing horizontal worksheet strip images into box crops."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from data._box_ops import (
    BLANK_INTENSITY_THRESH,
    BLANK_MIN_ACTIVE_PIXELS,
    BLANK_MIN_INK_MASS,
    MAX_BYTE_VAL,
    MIN_BORDER_STRIP_SIDE,
    POLARITY_MIDPOINT,
    compute_border_trims,
    compute_side_trim,
    find_contiguous_runs,
    hwc_to_float_gray,
    normalize_array_to_hwc,
    pil_to_detached_clean,
)

_STRIP_MIN_DIM = 4
_STRIP_OUTER_HORIZ_FRAC = 0.20
_STRIP_HORIZ_OCC_THRESH = 0.35
_STRIP_GUIDE_MIN_OCC = 0.70
_STRIP_GUIDE_SOLID_OCC = 0.85
_STRIP_GAP_MAX_OCC = 0.15
_STRIP_VERT_OCC_THRESH = 0.75
_STRIP_VERT_ONLY_OCC_THRESH = 0.90
_STRIP_COL_INK_THRESH = 0.01
_STRIP_VALLEY_PIXEL_THRESH = 0.12
_STRIP_BOX_HORIZ_LOCAL_OCC = 0.70
_STRIP_BOX_HORIZ_GLOBAL_OCC = 0.50
_OUTER_FRAME_X_FRAC = 0.15


def strip_input_to_pil(strip_img: Any) -> Image.Image:
    """Validate strip input and convert it to a clean detached RGB or L PIL Image."""
    if isinstance(strip_img, Image.Image):
        if strip_img.height < _STRIP_MIN_DIM or strip_img.width < _STRIP_MIN_DIM:
            raise ValueError("Strip image must be at least 4x4 pixels.")
        return pil_to_detached_clean(strip_img, prefer_rgb=True)
    if isinstance(strip_img, np.ndarray):
        hwc = normalize_array_to_hwc(strip_img, is_tensor=False)
        if hwc.shape[0] < _STRIP_MIN_DIM or hwc.shape[1] < _STRIP_MIN_DIM:
            raise ValueError("Strip image must be at least 4x4 pixels.")
        if hwc.ndim == 2 or hwc.shape[2] in (1, 2):
            return Image.fromarray(np.rint(hwc_to_float_gray(hwc)).astype(np.uint8))
        if hwc.shape[2] == 4:
            alpha = np.clip(hwc[:, :, 3:4] / MAX_BYTE_VAL, 0.0, 1.0)
            rgb = hwc[:, :, :3] * alpha + MAX_BYTE_VAL * (1.0 - alpha)
            return Image.fromarray(np.rint(np.clip(rgb, 0.0, MAX_BYTE_VAL)).astype(np.uint8))
        return Image.fromarray(np.rint(hwc[:, :, :3]).astype(np.uint8))
    raise ValueError("strip_img must be a PIL.Image.Image or numpy.ndarray.")


def strip_polarity_map(gray: np.ndarray) -> np.ndarray:
    """Build a polarity-normalized [0, 1] ink map (ink near 1.0, paper near 0.0)."""
    bg_ref = float(np.median(gray))
    work = (MAX_BYTE_VAL - gray) if bg_ref > POLARITY_MIDPOINT else gray
    cutoff = float(np.percentile(work, 70))
    bg_pixels = work[work <= cutoff]
    bg_val = float(np.median(bg_pixels)) if bg_pixels.size > 0 else float(np.median(work))
    denom = max(MAX_BYTE_VAL - bg_val, 1.0)
    return np.clip((work - bg_val) / denom, 0.0, 1.0).astype(np.float32)


def _filter_guide_rows(row_occ: np.ndarray, rows: list[int], from_top: bool) -> list[int]:
    if not rows:
        return []
    peak = float(row_occ[rows].max())
    if peak >= _STRIP_GUIDE_SOLID_OCC:
        return rows
    if peak < _STRIP_GUIDE_MIN_OCC:
        return []
    check_row = rows[-1] + 1 if from_top else rows[0] - 1
    if 0 <= check_row < row_occ.shape[0] and float(row_occ[check_row]) <= _STRIP_GAP_MAX_OCC:
        return rows
    return []


def detect_strip_grid(
    active: np.ndarray,
) -> tuple[int, int, list[int], list[int], list[tuple[int, int]]]:
    """Locate outer horizontal border lines and vertical grid dividers."""
    height, _ = active.shape
    row_occ = active.mean(axis=1)
    band_len = max(1, round(_STRIP_OUTER_HORIZ_FRAC * height))
    top_lim = min(band_len, height // 2)
    bot_start = max(top_lim, height - band_len)
    cand_top = [r for r in range(top_lim) if float(row_occ[r]) >= _STRIP_HORIZ_OCC_THRESH]
    cand_bot = [r for r in range(bot_start, height) if float(row_occ[r]) >= _STRIP_HORIZ_OCC_THRESH]
    y0 = cand_top[-1] + 1 if cand_top else 0
    y1 = cand_bot[0] if cand_bot else height
    if y1 <= y0:
        y0, y1 = 0, height
    col_occ = active[y0:y1, :].mean(axis=0)
    div_runs = find_contiguous_runs(np.where(col_occ >= _STRIP_VERT_OCC_THRESH)[0])
    has_horiz = bool(cand_top or cand_bot)
    thin_full = (
        len(div_runs) >= 2
        and all((e - s) <= max(4, height // 8) and float(col_occ[s:e].mean()) >= _STRIP_VERT_ONLY_OCC_THRESH for s, e in div_runs)
    )
    if div_runs and (has_horiz or thin_full):
        return y0, y1, cand_top, cand_bot, div_runs
    top_rows = _filter_guide_rows(row_occ, cand_top, from_top=True)
    bot_rows = _filter_guide_rows(row_occ, cand_bot, from_top=False)
    y_start = top_rows[-1] + 1 if top_rows else 0
    y_end = bot_rows[0] if bot_rows else height
    if y_end <= y_start:
        y_start, y_end = 0, height
    return y_start, y_end, top_rows, bot_rows, []


def _is_real_box_interval(
    active: np.ndarray,
    norm: np.ndarray,
    x0: int,
    x1: int,
    y_start: int,
    y_end: int,
    top_rows: list[int],
    bot_rows: list[int],
) -> bool:
    if x1 - x0 < 2:
        return False
    if top_rows and any(float(active[r, x0:x1].mean()) >= _STRIP_BOX_HORIZ_GLOBAL_OCC for r in top_rows):
        return True
    if bot_rows and any(float(active[r, x0:x1].mean()) >= _STRIP_BOX_HORIZ_GLOBAL_OCC for r in bot_rows):
        return True
    height = active.shape[0]
    local_band = max(1, round(_STRIP_OUTER_HORIZ_FRAC * height))
    top_local = active[: min(local_band, height // 2), x0:x1].mean(axis=1)
    bot_local = active[max(local_band, height - local_band) :, x0:x1].mean(axis=1)
    if (top_local.size > 0 and float(top_local.max()) >= _STRIP_BOX_HORIZ_LOCAL_OCC) or (
        bot_local.size > 0 and float(bot_local.max()) >= _STRIP_BOX_HORIZ_LOCAL_OCC
    ):
        return True
    patch = norm[y_start:y_end, x0:x1]
    ink_mask = patch > BLANK_INTENSITY_THRESH
    ink_count = int(ink_mask.sum())
    ink_mass = float(patch[ink_mask].sum()) if ink_count > 0 else 0.0
    return ink_count >= BLANK_MIN_ACTIVE_PIXELS and ink_mass >= BLANK_MIN_INK_MASS


def select_real_boxes(
    active: np.ndarray,
    norm: np.ndarray,
    y_start: int,
    y_end: int,
    top_rows: list[int],
    bot_rows: list[int],
    div_runs: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Filter candidate intervals between vertical dividers, dropping blank gutters/margins."""
    if not div_runs:
        return []
    width = active.shape[1]
    edges: list[tuple[int, int]] = [(0, div_runs[0][0])]
    for idx in range(len(div_runs) - 1):
        edges.append((div_runs[idx][1], div_runs[idx + 1][0]))
    edges.append((div_runs[-1][1], width))
    return [
        (x0, x1)
        for x0, x1 in edges
        if _is_real_box_interval(active, norm, x0, x1, y_start, y_end, top_rows, bot_rows)
    ]


def trim_outer_frame_x(
    active: np.ndarray,
    y_start: int,
    y_end: int,
    div_runs: list[tuple[int, int]],
    grid_boxes: list[tuple[int, int]],
) -> tuple[int, int]:
    """Strip outer left/right frame lines and margins when falling back to splitting/runs."""
    width = active.shape[1]
    if grid_boxes:
        return grid_boxes[0][0], grid_boxes[-1][1]
    if len(div_runs) >= 2:
        return div_runs[0][1], div_runs[-1][0]
    x_start, x_end = 0, width
    if len(div_runs) == 1:
        d0, d1 = div_runs[0]
        if d0 <= int(_OUTER_FRAME_X_FRAC * width):
            x_start = d1
        elif d1 >= width - int(_OUTER_FRAME_X_FRAC * width):
            x_end = d0
    if x_end - x_start >= MIN_BORDER_STRIP_SIDE:
        band_mask = active[y_start:y_end, x_start:x_end]
        left_trim = compute_side_trim(band_mask.T)
        right_trim = compute_side_trim(band_mask.T[::-1, :])
        if x_start + left_trim < x_end - right_trim:
            x_start += left_trim
            x_end -= right_trim
    return x_start, x_end


def detect_whitespace_runs(trimmed_norm: np.ndarray, x_start: int) -> list[tuple[int, int]]:
    """Split a borderless strip into character runs separated by whitespace valleys."""
    band_h, band_w = trimmed_norm.shape
    col_ink = (trimmed_norm > _STRIP_VALLEY_PIXEL_THRESH).mean(axis=0)
    raw_runs = find_contiguous_runs(np.where(col_ink > _STRIP_COL_INK_THRESH)[0])
    if not raw_runs:
        return []
    merge_gap = max(2, band_h // 18)
    merged: list[tuple[int, int]] = [raw_runs[0]]
    for start, end in raw_runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    pad = max(2, band_h // 8)
    padded: list[tuple[int, int]] = []
    for idx, (start, end) in enumerate(merged):
        left_bound = 0 if idx == 0 else (merged[idx - 1][1] + start) // 2
        right_bound = band_w if idx == len(merged) - 1 else (end + merged[idx + 1][0]) // 2
        cx0 = x_start + max(left_bound, start - pad)
        cx1 = x_start + min(right_bound, end + pad)
        padded.append((cx0, max(cx0 + 1, cx1)))
    return padded


def crop_box_clean(
    clean_pil: Image.Image,
    gray: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> Image.Image:
    """Crop a strip interval to a copied PIL Image and strip residual borders."""
    patch_gray = gray[y0:y1, x0:x1]
    top, bottom, left, right = compute_border_trims(patch_gray)
    cy0, cy1 = (y0 + top, y1 - bottom) if (y1 - bottom > y0 + top) else (y0, y1)
    cx0, cx1 = (x0 + left, x1 - right) if (x1 - right > x0 + left) else (x0, x1)
    return clean_pil.crop((cx0, cy0, cx1, cy1)).copy()

"""Spec-driven tests for data.preprocess and data public utilities."""

from __future__ import annotations

import errno
import io
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

import data
from data.preprocess import (
    CANONICAL_36_CHARS,
    EMNIST_BALANCED_CHARS,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_DIM,
    fix_emnist_orientation,
    fold_47_to_36_probs,
    load_validated_image_bytes,
    load_validated_image_file,
    preprocess_box,
    slice_strip_image,
)

PAIRED_LOWER = ("a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t")


def _make_28_glyph(letter: str) -> np.ndarray:
    arr = np.zeros((28, 28), dtype=np.float32)
    arr[4:24, 6:9] = 1.0
    if letter == "F":
        arr[4:7, 6:19] = 1.0
        arr[12:15, 6:15] = 1.0
    elif letter == "L":
        arr[21:24, 6:19] = 1.0
    return arr


def _draw_glyph(
    draw: ImageDraw.ImageDraw,
    letter: str,
    box: tuple[int, int, int, int],
    fill: int | tuple[int, ...],
    width: int = 4,
) -> None:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    strokes = {
        "L": [[(x0, y0), (x0, y1), (x1, y1)]],
        "F": [[(x0, y1), (x0, y0), (x1, y0)], [(x0, cy), (x0 + (x1 - x0) * 2 // 3, cy)]],
        "I": [[(cx, y0), (cx, y1)]],
        "A": [
            [(cx, y0), (x0, y1)],
            [(cx, y0), (x1, y1)],
            [(x0 + (x1 - x0) // 4, cy + 2), (x1 - (x1 - x0) // 4, cy + 2)],
        ],
        "M": [[(x0, y1), (x0, y0), (cx, cy), (x1, y0), (x1, y1)]],
        "C": [[(x1, y0), (x0, y0), (x0, y1), (x1, y1)]],
        "T": [[(x0, y0), (x1, y0)], [(cx, y0), (cx, y1)]],
    }
    for pts in strokes[letter]:
        draw.line(pts, fill=fill, width=width)


def _img_bytes(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _crafted_png_dims(width: int, height: int) -> bytes:
    ihdr_body = b"IHDR" + struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = struct.pack(">I", 13) + ihdr_body + struct.pack(">I", zlib.crc32(ihdr_body) & 0xFFFFFFFF)
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + ihdr + iend


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.flatten().float(), b.flatten().float()
    denom = float(af.norm() * bf.norm())
    return float(torch.dot(af, bf) / denom) if denom > 0 else 0.0


def _center_of_mass(t: torch.Tensor) -> tuple[float, float]:
    arr = t.squeeze(0).numpy()
    ys, xs = np.indices(arr.shape)
    total = float(arr.sum())
    return float((ys * arr).sum() / total), float((xs * arr).sum() / total)


def test_public_exports_and_class_mappings() -> None:
    expected_47 = tuple("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")
    expected_36 = tuple("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert isinstance(EMNIST_BALANCED_CHARS, tuple) and EMNIST_BALANCED_CHARS == expected_47
    assert isinstance(CANONICAL_36_CHARS, tuple) and CANONICAL_36_CHARS == expected_36
    assert MAX_IMAGE_BYTES == 5 * 1024 * 1024 and MAX_IMAGE_DIM == 4096
    assert data.EMNIST_BALANCED_CHARS == expected_47 and data.CANONICAL_36_CHARS == expected_36
    assert data.fold_47_to_36_probs is fold_47_to_36_probs and data.preprocess_box is preprocess_box


def test_fold_merges_all_11_lowercase_pairs_and_avoids_top2_duplicates() -> None:
    probs = np.zeros((47,), dtype=np.float64)
    probs[0] = 0.12
    for offset, ch in enumerate(PAIRED_LOWER):
        probs[EMNIST_BALANCED_CHARS.index(ch.upper())] = 0.02 + 0.002 * offset
        probs[EMNIST_BALANCED_CHARS.index(ch)] = 0.03 + 0.001 * offset
    folded = fold_47_to_36_probs(probs)
    expected = probs[:36].copy()
    for ch in PAIRED_LOWER:
        expected[CANONICAL_36_CHARS.index(ch.upper())] += probs[EMNIST_BALANCED_CHARS.index(ch)]
    expected /= expected.sum()
    assert np.isclose(folded.sum(), 1.0, atol=1e-6) and np.allclose(folded, expected, atol=1e-6)

    dup = torch.zeros((47,), dtype=torch.float32)
    dup[EMNIST_BALANCED_CHARS.index("A")] = 0.45
    dup[EMNIST_BALANCED_CHARS.index("a")] = 0.35
    dup[EMNIST_BALANCED_CHARS.index("B")] = 0.20
    top2 = torch.topk(fold_47_to_36_probs(dup), k=2).indices.tolist()
    assert [CANONICAL_36_CHARS[i] for i in top2] == ["A", "B"]


def test_fold_1d_and_2d_tensor_and_ndarray_normalize_without_mutation() -> None:
    vec = np.arange(1, 48, dtype=np.float32) * 2.0
    batch = np.stack([vec, vec * 3.0], axis=0)
    for arr in (vec, batch):
        orig_np = arr.copy()
        out_np = fold_47_to_36_probs(arr)
        assert isinstance(out_np, np.ndarray) and out_np.shape == (*arr.shape[:-1], 36)
        assert np.allclose(out_np.sum(axis=-1), 1.0, atol=1e-6) and np.array_equal(arr, orig_np)

        t_in = torch.from_numpy(arr.copy())
        t_orig = t_in.clone()
        out_t = fold_47_to_36_probs(t_in)
        assert isinstance(out_t, torch.Tensor) and out_t.shape == (*t_in.shape[:-1], 36)
        assert torch.allclose(out_t.sum(dim=-1), torch.ones(out_t.shape[:-1]), atol=1e-6)
        assert torch.equal(t_in, t_orig)


def test_fold_zero_sum_safe_and_rejects_invalid_inputs() -> None:
    z_np = np.zeros((2, 47), dtype=np.float32)
    z_np[1, 5] = 4.0
    out_np = fold_47_to_36_probs(z_np)
    assert np.isfinite(out_np).all() and (out_np >= 0).all() and np.isclose(out_np[1].sum(), 1.0, atol=1e-6)
    out_t = fold_47_to_36_probs(torch.zeros((47,), dtype=torch.float32))
    assert torch.isfinite(out_t).all() and (out_t >= 0).all()

    for bad_shape in ((36,), (48,), (2, 36), (1, 1, 47), ()):
        for bad_in in (np.ones(bad_shape, dtype=np.float32), torch.ones(bad_shape, dtype=torch.float32)):
            with pytest.raises(ValueError):
                fold_47_to_36_probs(bad_in)

    neg = np.ones((47,), dtype=np.float32)
    neg[3] = -0.1
    nan_arr = np.ones((2, 47), dtype=np.float32)
    nan_arr[1, 10] = np.nan
    for bad in (neg, torch.from_numpy(neg.copy()), nan_arr, torch.from_numpy(nan_arr.copy())):
        with pytest.raises(ValueError):
            fold_47_to_36_probs(bad)


def test_fix_emnist_orientation_matches_rot90_plus_hflip_across_dims() -> None:
    f_up, l_up = _make_28_glyph("F"), _make_28_glyph("L")
    f_raw, l_raw = f_up.T.copy(), l_up.T.copy()
    assert np.array_equal(np.flip(np.rot90(f_raw, k=-1, axes=(-2, -1)), axis=-1), f_up)

    batch_raw = np.stack([f_raw[None, ...], l_raw[None, ...]], axis=0)
    batch_exp = np.stack([f_up[None, ...], l_up[None, ...]], axis=0)
    for arr, expected in ((f_raw, f_up), (f_raw[None, ...], f_up[None, ...]), (batch_raw, batch_exp)):
        orig = arr.copy()
        out = fix_emnist_orientation(arr)
        assert isinstance(out, np.ndarray) and np.array_equal(out, expected)
        assert not np.array_equal(out, np.flip(expected, axis=-2))
        assert not np.array_equal(out, np.flip(expected, axis=-1))
        assert np.array_equal(arr, orig)

        t_in = torch.from_numpy(arr.copy())
        t_orig = t_in.clone()
        t_out = fix_emnist_orientation(t_in)
        assert isinstance(t_out, torch.Tensor) and torch.equal(t_out, torch.from_numpy(expected))
        assert torch.equal(t_in, t_orig)


def test_fix_emnist_orientation_rejects_low_dim_and_wrong_types() -> None:
    for bad in (np.zeros((28,)), np.array(1.0), torch.zeros((28,)), torch.tensor(1.0)):
        with pytest.raises(ValueError):
            fix_emnist_orientation(bad)
    for bad_type in ([[1.0, 2.0], [3.0, 4.0]], "bad", None):
        with pytest.raises((TypeError, ValueError)):
            fix_emnist_orientation(bad_type)


def test_preprocess_rgba_transparent_offcenter_centers_and_no_mutation() -> None:
    img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    _draw_glyph(ImageDraw.Draw(img), "L", (12, 12, 46, 54), (0, 0, 0, 255), width=7)
    arr_in = np.array(img, copy=True)
    arr_orig, pixels_orig = arr_in.copy(), img.tobytes()

    t, is_empty = preprocess_box(img)
    assert is_empty is False and isinstance(t, torch.Tensor)
    assert t.shape == (1, 28, 28) and t.dtype == torch.float32
    assert 0.0 <= float(t.min()) and 0.9 <= float(t.max()) <= 1.0
    assert all(float(t[0, r, c]) <= 1e-6 for r, c in ((0, 0), (0, 27), (27, 0), (27, 27)))
    cy, cx = _center_of_mass(t)
    assert abs(cy - 13.5) <= 1.5 and abs(cx - 13.5) <= 1.5 and img.tobytes() == pixels_orig

    t_arr, empty_arr = preprocess_box(arr_in)
    assert empty_arr is False and torch.allclose(t, t_arr, atol=1e-5) and np.array_equal(arr_in, arr_orig)


def test_preprocess_preserves_aspect_ratio_and_thin_highres_stroke() -> None:
    tall = Image.new("L", (140, 140), 255)
    _draw_glyph(ImageDraw.Draw(tall), "I", (60, 15, 80, 125), 0, width=8)
    t_tall, empty_tall = preprocess_box(tall)
    assert empty_tall is False
    ys, xs = torch.where(t_tall.squeeze(0) > 0.2)
    assert int(ys.max() - ys.min() + 1) >= 16
    assert int(ys.max() - ys.min() + 1) > 2 * int(xs.max() - xs.min() + 1)

    for stroke_w in (1, 2):
        thin = Image.new("L", (256, 256), 255)
        _draw_glyph(ImageDraw.Draw(thin), "L", (50, 50, 200, 200), 0, width=stroke_w)
        t_thin, empty_thin = preprocess_box(thin)
        assert empty_thin is False and float(t_thin.max()) >= 0.9
        assert float(t_thin[0, 18:26, 9:19].max(dim=0).values.min()) >= 0.4


def test_preprocess_polarity_parity_dark_on_white_vs_white_on_black() -> None:
    dark_on_white = Image.new("RGB", (128, 128), (255, 255, 255))
    _draw_glyph(ImageDraw.Draw(dark_on_white), "F", (36, 24, 92, 104), (0, 0, 0), width=8)
    white_on_black = Image.fromarray(255 - np.array(dark_on_white))

    t_dw, empty_dw = preprocess_box(dark_on_white)
    t_wb, empty_wb = preprocess_box(white_on_black)
    assert empty_dw is False and empty_wb is False
    assert float(t_dw[0, 0, 0]) <= 0.05 and float(t_wb[0, 0, 0]) <= 0.05
    assert float(t_dw.max()) >= 0.9 and float(t_wb.max()) >= 0.9
    assert _cos_sim(t_dw, t_wb) >= 0.95 and float((t_dw - t_wb).abs().mean()) <= 0.05


def test_preprocess_strips_flush_and_inset_borders_for_glyph() -> None:
    clean = Image.new("L", (64, 64), 255)
    _draw_glyph(ImageDraw.Draw(clean), "C", (18, 16, 46, 48), 0, width=5)
    t_clean, empty_clean = preprocess_box(clean)
    assert empty_clean is False
    ring = np.ones((28, 28), dtype=bool)
    ring[2:26, 2:26] = False

    for rect in ([0, 0, 63, 63], [2, 2, 61, 61]):
        boxed = clean.copy()
        ImageDraw.Draw(boxed).rectangle(rect, outline=0, width=2)
        t_box, empty_box = preprocess_box(boxed)
        assert empty_box is False and float(t_box[0].numpy()[ring].max()) <= 1e-6
        assert _cos_sim(t_box, t_clean) >= 0.95 and float((t_box - t_clean).abs().mean()) <= 0.05


def test_preprocess_blank_and_bordered_empty_boxes_return_zeros_and_true() -> None:
    flush_empty = Image.new("L", (64, 64), 255)
    ImageDraw.Draw(flush_empty).rectangle([0, 0, 63, 63], outline=0, width=2)
    inset_empty = Image.new("L", (64, 64), 255)
    ImageDraw.Draw(inset_empty).rectangle([2, 2, 61, 61], outline=0, width=2)
    samples = (
        flush_empty,
        inset_empty,
        Image.new("RGB", (64, 64), (255, 255, 255)),
        Image.new("RGB", (64, 64), (0, 0, 0)),
        Image.new("RGBA", (64, 64), (0, 0, 0, 0)),
        Image.new("RGB", (64, 64), (128, 128, 128)),
    )
    zeros = torch.zeros((1, 28, 28), dtype=torch.float32)
    for sample in samples:
        t, is_empty = preprocess_box(sample)
        assert is_empty is True and t.shape == (1, 28, 28) and t.dtype == torch.float32
        assert torch.equal(t, zeros)


def test_slice_strip_touching_grid_with_empty_middle_box() -> None:
    strip = Image.new("L", (256, 64), 255)
    draw = ImageDraw.Draw(strip)
    draw.rectangle([0, 0, 255, 63], outline=0, width=2)
    for x in (64, 128, 192):
        draw.line([(x, 0), (x, 63)], fill=0, width=2)
    for idx, ch in ((0, "C"), (2, "A"), (3, "T")):
        _draw_glyph(draw, ch, (idx * 64 + 16, 14, idx * 64 + 48, 50), 0, width=5)

    boxes = slice_strip_image(strip)
    assert len(boxes) == 4 and all(isinstance(b, Image.Image) for b in boxes)
    assert [preprocess_box(b)[1] for b in boxes] == [False, True, False, False]


def test_slice_strip_separated_boxes_with_white_gutters_no_phantoms() -> None:
    strip = Image.new("L", (252, 84), 255)
    draw = ImageDraw.Draw(strip)
    for idx, ch in enumerate(("C", "A", "T")):
        x0 = 10 + idx * 84
        draw.rectangle([x0, 10, x0 + 63, 73], outline=0, width=2)
        _draw_glyph(draw, ch, (x0 + 16, 24, x0 + 48, 60), 0, width=5)

    boxes = slice_strip_image(strip)
    assert len(boxes) == 3 and [preprocess_box(b)[1] for b in boxes] == [False, False, False]


def test_slice_strip_borderless_i_am_splits_by_valleys_without_splitting_i() -> None:
    strip = Image.new("L", (176, 64), 255)
    draw = ImageDraw.Draw(strip)
    _draw_glyph(draw, "I", (18, 12, 24, 52), 0, width=6)
    _draw_glyph(draw, "A", (56, 12, 88, 52), 0, width=5)
    _draw_glyph(draw, "M", (116, 12, 152, 52), 0, width=5)

    boxes = slice_strip_image(strip, num_boxes=None)
    assert len(boxes) == 3
    for box, ch, w, sw in zip(boxes, ("I", "A", "M"), (6, 32, 36), (6, 5, 5)):
        ref = Image.new("L", (64, 64), 255)
        _draw_glyph(ImageDraw.Draw(ref), ch, (14, 12, 14 + w, 52), 0, width=sw)
        t_box, empty_box = preprocess_box(box)
        assert empty_box is False and _cos_sim(t_box, preprocess_box(ref)[0]) >= 0.90


def test_slice_strip_explicit_num_boxes_slices_equally_including_trailing_blank() -> None:
    strip = Image.new("L", (256, 64), 255)
    draw = ImageDraw.Draw(strip)
    for idx, ch in enumerate(("C", "A", "T")):
        _draw_glyph(draw, ch, (idx * 64 + 16, 14, idx * 64 + 48, 50), 0, width=5)

    boxes = slice_strip_image(strip, num_boxes=4)
    assert len(boxes) == 4 and [preprocess_box(b)[1] for b in boxes] == [False, False, False, True]


@pytest.mark.parametrize("bad_n", [0, -1, True])
def test_slice_strip_rejects_invalid_num_boxes_with_value_error(bad_n: int | bool) -> None:
    strip = Image.new("L", (64, 64), 255)
    with pytest.raises(ValueError):
        slice_strip_image(strip, num_boxes=bad_n)


def test_load_validated_image_bytes_accepts_png_and_jpeg_and_checks_bounds() -> None:
    rgb = Image.new("RGB", (32, 24), (10, 20, 30))
    for fmt in ("PNG", "JPEG"):
        loaded = load_validated_image_bytes(_img_bytes(rgb, fmt))
        assert isinstance(loaded, Image.Image) and loaded.size == (32, 24)

    ok_png = _img_bytes(Image.new("RGB", (64, 64), (255, 255, 255)), "PNG")
    assert load_validated_image_bytes(ok_png, max_bytes=len(ok_png), max_dim=64).size == (64, 64)
    with pytest.raises(ValueError):
        load_validated_image_bytes(_img_bytes(Image.new("RGB", (65, 64), (255, 255, 255)), "PNG"), max_dim=64)


def test_load_validated_image_bytes_rejects_empty_oversize_bad_magic_and_dims() -> None:
    img = Image.new("RGB", (16, 16), (0, 0, 0))
    oversized = b"\x89PNG\r\n\x1a\n" + b"\x00" * (MAX_IMAGE_BYTES - 7)
    bad_payloads = (
        b"",
        oversized,
        _img_bytes(img, "GIF"),
        _img_bytes(img, "BMP"),
        b'<svg xmlns="http://www.w3.org/2000/svg"></svg>',
        b"plain text payload",
        _crafted_png_dims(5000, 5000),
        _crafted_png_dims(50000, 50000),
    )
    for payload in bad_payloads:
        with pytest.raises(ValueError):
            load_validated_image_bytes(payload)


def test_load_validated_image_file_valid_and_oversized(tmp_path: Path) -> None:
    png_path = tmp_path / "valid.png"
    png_bytes = _img_bytes(Image.new("RGB", (40, 40), (255, 255, 255)), "PNG")
    png_path.write_bytes(png_bytes)
    assert load_validated_image_file(png_path).size == (40, 40)

    with pytest.raises(ValueError) as size_exc:
        load_validated_image_file(png_path, max_bytes=len(png_bytes) - 1)
    assert str(size_exc.value) and "\n" not in str(size_exc.value) and "Traceback" not in str(size_exc.value)


def test_load_validated_image_file_missing_and_directory_errors(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError) as dir_exc:
        load_validated_image_file(tmp_path)
    assert str(dir_exc.value) and "\n" not in str(dir_exc.value) and "Traceback" not in str(dir_exc.value)

    with pytest.raises(FileNotFoundError) as missing_exc:
        load_validated_image_file(tmp_path / "missing.png")
    msg = str(missing_exc.value)
    assert msg and "\n" not in msg and "Traceback" not in msg
    assert missing_exc.value.errno == errno.ENOENT

"""The one path from a worksheet item (or its boxed strip image, or its box images) to per-box 36-class probabilities.

Every consumer (classical fusion tuning, BERT tuning, the benchmark, the app) goes through
box images -> preprocess_box -> CNN -> softmax -> fold_47_to_36_probs, with strips first sliced by
slice_strip_image, so tuning, evaluation and the app see exactly the same tensors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from data.prepare_data import render_worksheet_item_strip
from data.preprocess import fold_47_to_36_probs, preprocess_box, slice_strip_image

__all__ = ["batch_to_probs", "boxes_to_tensors", "item_to_box_probs", "item_to_strip", "strip_to_box_tensors", "strip_to_probs"]


def item_to_strip(item: Mapping[str, Any], emnist_ds: Any) -> Image.Image:
    """Render the boxed strip image a student would have written for this worksheet item."""
    return render_worksheet_item_strip(item, emnist_ds)


def boxes_to_tensors(boxes: Sequence[Image.Image]) -> tuple[torch.Tensor, list[bool]]:
    """Preprocess box images into a (N, 1, 28, 28) float32 batch plus a per-box 'empty' flag."""
    if not boxes:
        raise ValueError("No character boxes to preprocess.")
    prepared = [preprocess_box(box) for box in boxes]
    return torch.stack([tensor for tensor, _ in prepared]), [bool(empty) for _, empty in prepared]


def batch_to_probs(batch: torch.Tensor, model: nn.Module) -> np.ndarray:
    """Run a character model on a box batch; return folded (N, 36) float64 probabilities."""
    with torch.inference_mode():
        probs_47 = torch.softmax(model(batch), dim=1)
    return np.asarray(fold_47_to_36_probs(probs_47).cpu().numpy(), dtype=np.float64)


def strip_to_box_tensors(strip: Image.Image | np.ndarray, num_boxes: int | None = None) -> torch.Tensor:
    """Slice a strip into boxes and preprocess each into a (N, 1, 28, 28) float32 batch."""
    boxes = slice_strip_image(strip, num_boxes=num_boxes)
    if not boxes:
        raise ValueError("No character boxes were found in the strip image.")
    return boxes_to_tensors(boxes)[0]


def strip_to_probs(strip: Image.Image | np.ndarray, model: nn.Module, num_boxes: int | None = None) -> np.ndarray:
    """Run the character model on every box of a strip; return folded (N, 36) float64 probabilities."""
    return batch_to_probs(strip_to_box_tensors(strip, num_boxes=num_boxes), model)


def item_to_box_probs(item: Mapping[str, Any], emnist_ds: Any, model: nn.Module) -> np.ndarray:
    """Render an item's strip, slice it into len(target_word) boxes, and return (n_chars, 36) probabilities."""
    target = str(item.get("target_word", ""))
    if not target:
        raise ValueError(f"Worksheet item {item.get('id')} has no target_word.")
    return strip_to_probs(item_to_strip(item, emnist_ds), model, num_boxes=len(target))

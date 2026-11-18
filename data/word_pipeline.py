"""The one path from a worksheet item (or its boxed strip image) to per-box 36-class probabilities.

Every consumer (classical fusion tuning, BERT tuning, the benchmark, the app) goes through
strip image -> slice_strip_image -> preprocess_box -> CNN -> softmax -> fold_47_to_36_probs,
so tuning and evaluation see exactly the boxes the app sees.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from data.prepare_data import render_worksheet_item_strip
from data.preprocess import fold_47_to_36_probs, preprocess_box, slice_strip_image

__all__ = ["item_to_box_probs", "item_to_strip", "strip_to_box_tensors", "strip_to_probs"]


def item_to_strip(item: Mapping[str, Any], emnist_ds: Any) -> Image.Image:
    """Render the boxed strip image a student would have written for this worksheet item."""
    return render_worksheet_item_strip(item, emnist_ds)


def strip_to_box_tensors(strip: Image.Image | np.ndarray, num_boxes: int | None = None) -> torch.Tensor:
    """Slice a strip into boxes and preprocess each into a (N, 1, 28, 28) float32 batch."""
    boxes = slice_strip_image(strip, num_boxes=num_boxes)
    if not boxes:
        raise ValueError("No character boxes were found in the strip image.")
    return torch.stack([preprocess_box(box)[0] for box in boxes])


def strip_to_probs(strip: Image.Image | np.ndarray, model: nn.Module, num_boxes: int | None = None) -> np.ndarray:
    """Run the character model on every box of a strip; return folded (N, 36) float64 probabilities."""
    batch = strip_to_box_tensors(strip, num_boxes=num_boxes)
    with torch.inference_mode():
        probs_47 = torch.softmax(model(batch), dim=1)
    return np.asarray(fold_47_to_36_probs(probs_47).cpu().numpy(), dtype=np.float64)


def item_to_box_probs(item: Mapping[str, Any], emnist_ds: Any, model: nn.Module) -> np.ndarray:
    """Render an item's strip, slice it into len(target_word) boxes, and return (n_chars, 36) probabilities."""
    target = str(item.get("target_word", ""))
    if not target:
        raise ValueError(f"Worksheet item {item.get('id')} has no target_word.")
    return strip_to_probs(item_to_strip(item, emnist_ds), model, num_boxes=len(target))

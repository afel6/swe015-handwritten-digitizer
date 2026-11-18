"""Character recognition architectures and FP32 checkpoint helpers."""

from __future__ import annotations

import errno
import math
import os
import pickle
import warnings
import zipfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from data._io_errors import map_os_error as _map_os_error

NUM_CLASSES: int = 47
INPUT_DIM: int = 784

_IMAGE_HEIGHT = 28
_IMAGE_WIDTH = 28
_MLP_HIDDEN1: int = 512
_MLP_HIDDEN2: int = 256
_CNN_C1: int = 32
_CNN_C2: int = 64
_CNN_C3: int = 128
_CNN_KERNEL_SIZE: int = 3
_CNN_FEATURE_SIDE: int = 7
_CNN_FC_HIDDEN: int = 256
_CNN_DROP1: float = 0.3
_CNN_DROP2: float = 0.2
_MAX_META_DEPTH: int = 32
_AO_QUANT_MODULE_PATTERN: str = r"^torch\.ao\.quantization(?:\.|$)"
_AO_QUANT_MSG_PATTERN: str = r".*torch\.ao\.quantization.*"
_LOAD_ERRORS = (
    EOFError, RuntimeError, pickle.UnpicklingError, zipfile.BadZipFile, ValueError, TypeError
)
_SCALAR_CASTS: tuple[tuple[type[Any], type[Any]], ...] = (
    (bool, bool), (int, int), (float, float), (str, str)
)

__all__ = [
    "INPUT_DIM",
    "NUM_CLASSES",
    "CharCNN",
    "DenseMLP",
    "load_char_cnn_fp32",
    "load_dense_mlp",
    "save_fp32_checkpoint",
]


def _validate_num_classes(num_classes: Any) -> int:
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be an integer >= 1.")
    return int(num_classes)


def _validate_dropout(dropout: Any) -> float:
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(float(dropout))
        or not (0.0 <= float(dropout) < 1.0)
    ):
        raise ValueError("dropout must be a finite real number in [0.0, 1.0).")
    return float(dropout)


def _parse_device(device: str | torch.device) -> torch.device:
    try:
        return torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid target device: {device!r}") from exc


def _sanitize_mapping(mapping: Mapping[str, Any], depth: int) -> dict[str, Any]:
    if depth > _MAX_META_DEPTH:
        raise ValueError("Checkpoint metadata nesting exceeds maximum depth.")
    cleaned: dict[str, Any] = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise TypeError("Checkpoint metadata keys must be strings.")
        cleaned[str(key)] = _to_pure_primitive(item, depth + 1)
    return cleaned


def _to_pure_primitive(value: Any, depth: int = 0) -> Any:
    if depth > _MAX_META_DEPTH:
        raise ValueError("Checkpoint metadata nesting exceeds maximum depth.")
    if value is None:
        return None
    for cls, cast in _SCALAR_CASTS:
        if isinstance(value, cls):
            return cast(value)
    if isinstance(value, tuple):
        return tuple(_to_pure_primitive(item, depth + 1) for item in value)
    if isinstance(value, list):
        return [_to_pure_primitive(item, depth + 1) for item in value]
    if isinstance(value, Mapping):
        return _sanitize_mapping(value, depth)
    raise ValueError(
        "Checkpoint metadata values must be pure primitives "
        "(str, int, float, bool, None, list, tuple, dict)."
    )


def _sanitize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None.")
    return _sanitize_mapping(metadata, depth=0)


def _make_conv_block(in_ch: int, out_ch: int, *, with_pool: bool) -> nn.Sequential:
    layers: list[tuple[str, nn.Module]] = [
        ("conv", nn.Conv2d(in_ch, out_ch, _CNN_KERNEL_SIZE, padding=1, bias=False)),
        ("bn", nn.BatchNorm2d(out_ch)),
        ("relu", nn.ReLU()),
    ]
    if with_pool:
        layers.append(("pool", nn.MaxPool2d(2)))
    return nn.Sequential(OrderedDict(layers))


class DenseMLP(nn.Module):
    """Three-layer feedforward classifier mapping 784 pixels to character logits."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.2) -> None:
        super().__init__()
        valid_classes = _validate_num_classes(num_classes)
        valid_dropout = _validate_dropout(dropout)
        self.num_classes = valid_classes
        self.dropout_rate = valid_dropout
        self.input_dim = INPUT_DIM
        self.image_height = _IMAGE_HEIGHT
        self.image_width = _IMAGE_WIDTH
        self.net = nn.Sequential(
            OrderedDict([
                ("flatten", nn.Flatten(start_dim=1)),
                ("fc1", nn.Linear(INPUT_DIM, _MLP_HIDDEN1)),
                ("relu1", nn.ReLU()),
                ("drop1", nn.Dropout(valid_dropout)),
                ("fc2", nn.Linear(_MLP_HIDDEN1, _MLP_HIDDEN2)),
                ("relu2", nn.ReLU()),
                ("drop2", nn.Dropout(valid_dropout)),
                ("fc3", nn.Linear(_MLP_HIDDEN2, valid_classes)),
            ])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for (B, 784) or (B, 1, 28, 28) floating-point tensors."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor.")
        if not x.is_floating_point():
            raise ValueError("Input x must be a floating-point tensor.")
        valid_2d = x.ndim == 2 and x.shape[1] == self.input_dim
        valid_4d = x.ndim == 4 and (x.shape[1], x.shape[2], x.shape[3]) == (
            1,
            self.image_height,
            self.image_width,
        )
        if not (valid_2d or valid_4d):
            raise ValueError(
                "Input x must have shape (B, 784) or (B, 1, 28, 28)."
            )
        return self.net(x)


class CharCNN(nn.Module):
    """Quantization-ready 3-block CNN preserving 7x7 spatial features for Grad-CAM."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        valid_classes = _validate_num_classes(num_classes)
        self.num_classes = valid_classes
        self.image_height = _IMAGE_HEIGHT
        self.image_width = _IMAGE_WIDTH
        self.quant = torch.ao.quantization.QuantStub()
        self.block1 = _make_conv_block(1, _CNN_C1, with_pool=True)
        self.block2 = _make_conv_block(_CNN_C1, _CNN_C2, with_pool=True)
        self.block3 = _make_conv_block(_CNN_C2, _CNN_C3, with_pool=False)
        flat_dim = _CNN_C3 * _CNN_FEATURE_SIDE * _CNN_FEATURE_SIDE
        self.classifier = nn.Sequential(
            OrderedDict([
                ("flatten", nn.Flatten(start_dim=1)),
                ("drop1", nn.Dropout(_CNN_DROP1)),
                ("fc1", nn.Linear(flat_dim, _CNN_FC_HIDDEN)),
                ("relu1", nn.ReLU()),
                ("drop2", nn.Dropout(_CNN_DROP2)),
                ("fc2", nn.Linear(_CNN_FC_HIDDEN, valid_classes)),
            ])
        )
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for (B, 1, 28, 28) floating-point tensors."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor.")
        if not x.is_floating_point():
            raise ValueError("Input x must be a floating-point tensor.")
        if x.ndim != 4 or (x.shape[1], x.shape[2], x.shape[3]) != (
            1,
            self.image_height,
            self.image_width,
        ):
            raise ValueError("Input x must have shape (B, 1, 28, 28).")
        out = self.quant(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.classifier(out)
        return self.dequant(out)

    def fuse_model(self) -> None:
        """Switch to eval mode and fuse Conv/BN/ReLU and Linear/ReLU blocks once."""
        self.eval()
        if isinstance(self.block1.bn, nn.Identity):
            return
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=DeprecationWarning, module=_AO_QUANT_MODULE_PATTERN
            )
            warnings.filterwarnings(
                "ignore", category=DeprecationWarning, message=_AO_QUANT_MSG_PATTERN
            )
            for block in (self.block1, self.block2, self.block3):
                torch.ao.quantization.fuse_modules(
                    block, ["conv", "bn", "relu"], inplace=True
                )
            torch.ao.quantization.fuse_modules(
                self.classifier, ["fc1", "relu1"], inplace=True
            )

    def target_gradcam_layer(self) -> nn.Module:
        """Return the final 7x7 convolutional activation layer without mutating self."""
        return (
            self.block3.conv
            if isinstance(self.block3.relu, nn.Identity)
            else self.block3.relu
        )


def _resolve_model_num_classes(model: nn.Module) -> int | None:
    attr = getattr(model, "num_classes", None)
    if attr is not None:
        return _validate_num_classes(attr)
    for submodule in reversed(list(model.modules())):
        if isinstance(submodule, nn.Linear):
            return _validate_num_classes(int(submodule.out_features))
    return None


def save_fp32_checkpoint(
    model: nn.Module,
    path: str | os.PathLike[str],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a CPU FP32 state_dict, num_classes, and primitive metadata to disk."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("Checkpoint path must be a string or path-like object.")
    path_str = os.fspath(path)
    if not path_str:
        raise ValueError("Checkpoint path must not be empty.")
    target = Path(path_str)
    if target.is_dir():
        raise IsADirectoryError(
            errno.EISDIR, f"Expected a checkpoint file path, got a directory: {path_str}"
        )
    raw_state = model.state_dict()
    if any(isinstance(v, torch.Tensor) and v.is_quantized for v in raw_state.values()) or any(
        mod.__class__.__module__.startswith("torch.ao.nn.quantized")
        for mod in model.modules()
    ):
        raise ValueError("save_fp32_checkpoint requires an unquantized FP32 model.")
    clean_metadata = _sanitize_metadata(metadata)
    cpu_state = {
        k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
        for k, v in raw_state.items()
    }
    payload: dict[str, Any] = {"state_dict": cpu_state}
    num_classes = _resolve_model_num_classes(model)
    if num_classes is not None:
        payload["num_classes"] = num_classes
    if clean_metadata is not None:
        payload["metadata"] = clean_metadata
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as handle:
            torch.save(payload, handle)
    except OSError as err:
        raise _map_os_error(err, path_str, "writing") from err
    return target


def _infer_num_classes_from_state(
    state_dict: Mapping[str, Any], head_weight_key: str
) -> int:
    head_weight = state_dict.get(head_weight_key)
    if isinstance(head_weight, torch.Tensor) and head_weight.ndim == 2 and head_weight.shape[0] >= 1:
        return int(head_weight.shape[0])
    return NUM_CLASSES


def _is_tensor_state_mapping(raw_state: Any) -> bool:
    return isinstance(raw_state, Mapping) and len(raw_state) > 0 and all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in raw_state.items()
    )


def _extract_checkpoint_parts(
    payload: Any, head_weight_key: str
) -> tuple[dict[str, torch.Tensor], int]:
    if not isinstance(payload, Mapping) or len(payload) == 0:
        raise ValueError(
            "Invalid checkpoint format: expected a state_dict or checkpoint mapping."
        )
    if "state_dict" in payload:
        raw_state = payload["state_dict"]
        if not _is_tensor_state_mapping(raw_state):
            raise ValueError(
                "Invalid checkpoint: 'state_dict' must be a non-empty str -> Tensor mapping."
            )
        num_classes = (
            _validate_num_classes(payload["num_classes"])
            if "num_classes" in payload
            else _infer_num_classes_from_state(raw_state, head_weight_key)
        )
    else:
        raw_state = payload
        if not _is_tensor_state_mapping(raw_state):
            raise ValueError(
                "Invalid checkpoint state_dict: expected non-empty str -> Tensor entries."
            )
        num_classes = _infer_num_classes_from_state(raw_state, head_weight_key)
    return dict(raw_state), num_classes


def _load_fp32_model(
    path: str | os.PathLike[str],
    device: str | torch.device,
    model_cls: type[DenseMLP | CharCNN],
    head_weight_key: str,
) -> Any:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("Checkpoint path must be a string or path-like object.")
    target_device = _parse_device(device)
    path_str = os.fspath(path)
    if not path_str:
        raise FileNotFoundError(errno.ENOENT, "Checkpoint file not found: ''")
    try:
        with open(path_str, "rb") as handle:
            payload = torch.load(handle, map_location="cpu", weights_only=True)
    except OSError as err:
        raise _map_os_error(err, path_str, "reading") from err
    except _LOAD_ERRORS as exc:
        raise ValueError(f"Invalid or corrupted checkpoint file: {path_str}") from exc

    state_dict, num_classes = _extract_checkpoint_parts(payload, head_weight_key)
    model = model_cls(num_classes=num_classes)
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(
            f"Checkpoint weights do not match {model_cls.__name__}: {path_str}"
        ) from exc
    return model.to(target_device).eval()


def load_dense_mlp(
    path: str | os.PathLike[str], device: str | torch.device = "cpu"
) -> DenseMLP:
    """Load an FP32 DenseMLP checkpoint onto the target device in eval mode."""
    return _load_fp32_model(path, device, DenseMLP, "net.fc3.weight")


def load_char_cnn_fp32(
    path: str | os.PathLike[str], device: str | torch.device = "cpu"
) -> CharCNN:
    """Load an FP32 CharCNN checkpoint onto the target device in eval mode."""
    return _load_fp32_model(path, device, CharCNN, "classifier.fc2.weight")

"""Static CNN INT8 PTQ, dynamic BERT INT8 quantization, checkpoint I/O, and CPU benchmarking."""

from __future__ import annotations

import copy
import errno
import io
import itertools
import os
import pickle
import stat
import time
import warnings
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models.char_models import NUM_CLASSES, CharCNN

__all__ = [
    "benchmark_cpu_latency",
    "get_model_disk_size_mb",
    "get_quantization_engine",
    "load_quantized_bert_dynamic",
    "load_quantized_cnn",
    "quantize_bert_dynamic",
    "quantize_cnn_static",
    "save_quantized_cnn",
    "save_quantized_state_dict",
]

_CNN_INPUT_SIZE = 28
_BYTES_PER_MB = 1024.0 * 1024.0
_MS_PER_SECOND = 1000.0
_ENGINE_PRIORITY: tuple[str, ...] = ("onednn", "fbgemm", "x86", "qnnpack")


@contextmanager
def _suppress_torch_quant_warnings() -> Iterator[None]:
    """Suppress only known PyTorch eager-quantization and TypedStorage warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*torch\.ao\.quantization is deprecated.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Default qconfig of oneDNN backend.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*TypedStorage is deprecated.*",
            category=UserWarning,
        )
        yield


def _supported_engines() -> list[str]:
    return [engine for engine in torch.backends.quantized.supported_engines if engine != "none"]


def _has_quantized_submodule(model: nn.Module) -> bool:
    return any(
        "quantized" in type(mod).__module__.lower()
        or "quantized" in type(mod).__name__.lower()
        for mod in model.modules()
    )


def _validate_eager_module(model: Any, label: str) -> None:
    if not isinstance(model, nn.Module) or isinstance(model, torch.jit.ScriptModule):
        raise TypeError(f"{label} must be an eager torch.nn.Module (not a ScriptModule).")


def _validate_path(path: Any) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be a string or os.PathLike object.")
    raw = os.fspath(path)
    if not isinstance(raw, str):
        raise TypeError("path must resolve to a string path.")
    if not raw:
        raise ValueError("path must not be empty.")
    return Path(raw)


def _map_os_error(err: OSError, path: Path, action: str) -> OSError:
    if err.errno == errno.ENOENT:
        return FileNotFoundError(errno.ENOENT, f"Checkpoint file not found: {path}")
    if err.errno == errno.EISDIR or (err.errno == errno.EACCES and path.is_dir()):
        return IsADirectoryError(
            errno.EISDIR, f"Expected a checkpoint file, got a directory: {path}"
        )
    if err.errno == errno.EACCES:
        return PermissionError(errno.EACCES, f"Permission denied to {action} file: {path}")
    return OSError(err.errno, f"Unable to {action} checkpoint file: {path}")


def _save_checkpoint(payload: dict[str, Any], path: str | os.PathLike[str]) -> Path:
    out_path = _validate_path(path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as handle:
            torch.save(payload, handle)
    except OSError as err:
        raise _map_os_error(err, out_path, action="write") from err
    return out_path


def _load_checkpoint(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    in_path = _validate_path(path)
    try:
        with open(in_path, "rb") as handle, _suppress_torch_quant_warnings():
            ckpt = torch.load(handle, map_location="cpu", weights_only=True)
    except OSError as err:
        raise _map_os_error(err, in_path, action="read") from err
    except (
        EOFError,
        RuntimeError,
        ValueError,
        TypeError,
        pickle.UnpicklingError,
        zipfile.BadZipFile,
    ) as exc:
        raise ValueError(f"Invalid or corrupted quantized checkpoint: {in_path}") from exc
    if (
        not isinstance(ckpt, Mapping)
        or "state_dict" not in ckpt
        or not isinstance(ckpt["state_dict"], Mapping)
    ):
        raise ValueError(f"Invalid quantized checkpoint: missing 'state_dict' in {in_path}.")
    return ckpt


def _infer_num_classes(model: nn.Module) -> int:
    classifier = getattr(model, "classifier", None)
    fc2 = getattr(classifier, "fc2", None) if classifier is not None else None
    out_features = getattr(fc2, "out_features", None) if fc2 is not None else None
    if isinstance(out_features, int) and not isinstance(out_features, bool) and out_features >= 1:
        return out_features
    return NUM_CLASSES


def _extract_calib_tensor(batch: Any) -> torch.Tensor:
    item = batch
    if isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise ValueError("Calibration batch tuple/list must not be empty.")
        item = batch[0]
    if not isinstance(item, torch.Tensor):
        raise TypeError("calib_loader must yield torch.Tensor batches or (inputs, ...) tuples.")
    return item.detach().to(device="cpu", dtype=torch.float32)


def get_quantization_engine(preferred: str | None = None) -> str:
    """Select an available PyTorch CPU quantization engine."""
    supported = _supported_engines()
    if preferred is not None:
        if not isinstance(preferred, str) or preferred not in supported:
            raise ValueError(
                f"Unsupported quantization engine {preferred!r}; supported engines: {supported}."
            )
        return preferred
    if not supported:
        raise RuntimeError("No supported PyTorch quantization engine is available.")
    for candidate in _ENGINE_PRIORITY:
        if candidate in supported:
            return candidate
    return supported[0]


def quantize_cnn_static(
    fp32_cnn: nn.Module,
    calib_loader: Iterable[Any],
    engine: str | None = None,
) -> nn.Module:
    """Quantize an FP32 CNN with eager static PTQ using calibration batches."""
    _validate_eager_module(fp32_cnn, "fp32_cnn")
    if not callable(getattr(fp32_cnn, "fuse_model", None)):
        raise TypeError("fp32_cnn must define a callable fuse_model() method.")
    if _has_quantized_submodule(fp32_cnn):
        raise ValueError("fp32_cnn already contains quantized submodules.")
    if calib_loader is None or isinstance(calib_loader, (str, bytes, bytearray, Mapping)):
        raise TypeError("calib_loader must be an iterable of calibration batches.")
    calib_iter = iter(calib_loader)

    chosen_engine = get_quantization_engine(engine)
    torch.backends.quantized.engine = chosen_engine

    fused = copy.deepcopy(fp32_cnn).cpu().eval()
    fused.fuse_model()

    with _suppress_torch_quant_warnings():
        fused.qconfig = torch.ao.quantization.get_default_qconfig(chosen_engine)
        prepared = torch.ao.quantization.prepare(fused, inplace=False)

    batch_count = 0
    with torch.inference_mode():
        for batch in calib_iter:
            prepared(_extract_calib_tensor(batch))
            batch_count += 1
    if batch_count == 0:
        raise ValueError("calib_loader must yield at least one batch for static calibration.")

    with _suppress_torch_quant_warnings():
        int8_model = torch.ao.quantization.convert(prepared, inplace=False)
    return int8_model.eval()


def save_quantized_cnn(int8_model: nn.Module, path: str | os.PathLike[str]) -> Path:
    """Save a static-quantized INT8 CNN checkpoint with engine and class-count metadata."""
    _validate_eager_module(int8_model, "int8_model")
    if not _has_quantized_submodule(int8_model):
        raise ValueError("int8_model must contain at least one quantized submodule.")
    payload = {
        "state_dict": int8_model.state_dict(),
        "engine": torch.backends.quantized.engine,
        "num_classes": _infer_num_classes(int8_model),
    }
    return _save_checkpoint(payload, path)


def load_quantized_cnn(
    path: str | os.PathLike[str],
    engine: str | None = None,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Reconstruct a static-quantized CharCNN from disk in CPU eval mode."""
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer.")
    if engine is not None:
        chosen_engine = get_quantization_engine(engine)
    else:
        chosen_engine = ""

    ckpt = _load_checkpoint(path)
    if not chosen_engine:
        supported = _supported_engines()
        ckpt_engine = ckpt.get("engine")
        if isinstance(ckpt_engine, str) and ckpt_engine in supported:
            chosen_engine = ckpt_engine
        else:
            chosen_engine = get_quantization_engine(None)
    torch.backends.quantized.engine = chosen_engine

    raw_classes = ckpt.get("num_classes")
    if raw_classes is None:
        resolved_classes = num_classes
    elif isinstance(raw_classes, bool) or not isinstance(raw_classes, int) or raw_classes < 1:
        raise ValueError("Invalid 'num_classes' metadata in quantized CNN checkpoint.")
    else:
        resolved_classes = raw_classes

    skeleton = CharCNN(num_classes=resolved_classes).cpu().eval()
    skeleton.fuse_model()
    with _suppress_torch_quant_warnings():
        skeleton.qconfig = torch.ao.quantization.get_default_qconfig(chosen_engine)
        prepared = torch.ao.quantization.prepare(skeleton, inplace=False)
        with torch.inference_mode():
            prepared(torch.zeros((1, 1, _CNN_INPUT_SIZE, _CNN_INPUT_SIZE), dtype=torch.float32))
        int8_model = torch.ao.quantization.convert(prepared, inplace=False)
    int8_model.load_state_dict(ckpt["state_dict"])
    return int8_model.eval()


def quantize_bert_dynamic(fp32_bert: nn.Module) -> nn.Module:
    """Return a CPU dynamic-INT8 quantized copy of a Transformer module."""
    _validate_eager_module(fp32_bert, "fp32_bert")
    if _has_quantized_submodule(fp32_bert):
        raise ValueError("fp32_bert already contains quantized submodules.")
    if torch.backends.quantized.engine not in _supported_engines():
        torch.backends.quantized.engine = get_quantization_engine(None)
    model_cpu = copy.deepcopy(fp32_bert).cpu().eval()
    with _suppress_torch_quant_warnings():
        int8_model = torch.ao.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear},
            dtype=torch.qint8,
        )
    return int8_model.eval()


def save_quantized_state_dict(
    model: nn.Module,
    path: str | os.PathLike[str],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a model's state_dict and optional metadata to disk."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    payload: dict[str, Any] = {}
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None.")
        if "state_dict" in metadata:
            raise ValueError("metadata must not contain reserved key 'state_dict'.")
        payload.update(metadata)
    payload["state_dict"] = model.state_dict()
    return _save_checkpoint(payload, path)


def load_quantized_bert_dynamic(
    bert_factory: Callable[[], nn.Module],
    path: str | os.PathLike[str],
) -> nn.Module:
    """Instantiate a skeleton BERT, apply dynamic INT8 conversion, and load weights."""
    if not callable(bert_factory):
        raise TypeError("bert_factory must be a callable returning an nn.Module.")
    ckpt = _load_checkpoint(path)
    skeleton = bert_factory()
    _validate_eager_module(skeleton, "bert_factory() result")
    int8_model = quantize_bert_dynamic(skeleton.cpu().eval())
    int8_model.load_state_dict(ckpt["state_dict"])
    return int8_model.eval()


def get_model_disk_size_mb(
    path_or_state: str | os.PathLike[str] | nn.Module | Mapping[str, Any],
) -> float:
    """Measure serialized checkpoint size in megabytes (MiB) from path, module, or mapping."""
    if isinstance(path_or_state, (str, os.PathLike)):
        target = _validate_path(path_or_state)
        try:
            st = os.stat(target)
        except OSError as err:
            raise _map_os_error(err, target, action="stat") from err
        if stat.S_ISDIR(st.st_mode):
            raise IsADirectoryError(
                errno.EISDIR, f"Expected a checkpoint file, got a directory: {target}"
            )
        return float(st.st_size) / _BYTES_PER_MB

    if isinstance(path_or_state, nn.Module):
        buf = io.BytesIO()
        torch.save(path_or_state.state_dict(), buf)
        return float(buf.getbuffer().nbytes) / _BYTES_PER_MB

    if isinstance(path_or_state, Mapping):
        buf = io.BytesIO()
        serializable = path_or_state if isinstance(path_or_state, dict) else dict(path_or_state)
        torch.save(serializable, buf)
        return float(buf.getbuffer().nbytes) / _BYTES_PER_MB

    raise TypeError("path_or_state must be a file path, torch.nn.Module, or state Mapping.")


def benchmark_cpu_latency(
    fn: Callable[[Any], Any],
    inputs_iter: Iterable[Any],
    warmup: int = 10,
    runs: int = 100,
) -> dict[str, float]:
    """Measure per-call CPU latency statistics in milliseconds inside inference_mode."""
    if not callable(fn):
        raise TypeError("fn must be callable.")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer.")
    if not isinstance(runs, int) or isinstance(runs, bool) or runs < 1:
        raise ValueError("runs must be a positive integer.")

    try:
        iterator = iter(inputs_iter)
    except TypeError as exc:
        raise TypeError("inputs_iter must be an iterable.") from exc

    collected = list(itertools.islice(iterator, warmup + runs))
    if not collected:
        raise ValueError("inputs_iter must yield at least one input item.")

    cycled = itertools.cycle(collected)
    latencies_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            fn(next(cycled))
        for _ in range(runs):
            item = next(cycled)
            start = time.perf_counter()
            fn(item)
            latencies_ms.append((time.perf_counter() - start) * _MS_PER_SECOND)

    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }

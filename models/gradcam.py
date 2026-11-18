"""FP32 Grad-CAM overlay generation and base64 PNG encoding."""

from __future__ import annotations

import base64
import io
import math

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

__all__ = ["generate_gradcam_b64"]

_INPUT_SHAPE: tuple[int, int, int, int] = (1, 1, 28, 28)
_OUTPUT_SIZE: tuple[int, int] = (28, 28)
_LOGITS_NDIM: int = 2
_FEATURE_MAP_NDIM: int = 4
_CAM_EPS: float = 1e-8
_GRAY_SPAN_EPS: float = 1e-8
_GRAY_BLEND_WEIGHT: float = 0.55
_HEAT_BLEND_WEIGHT: float = 0.45
_RGB_CHANNELS: int = 3
_UINT8_MAX: float = 255.0
_JET_COLORMAP_NAME: str = "jet"


def _validate_fp32_cnn(fp32_cnn: nn.Module) -> nn.Module:
    """Validate an eager FP32 nn.Module and return its target Grad-CAM layer."""
    if not isinstance(fp32_cnn, nn.Module):
        raise TypeError("fp32_cnn must be a torch.nn.Module.")
    if isinstance(fp32_cnn, torch.jit.ScriptModule):
        raise TypeError("fp32_cnn must be an eager FP32 nn.Module, not a ScriptModule.")
    for mod in fp32_cnn.modules():
        if isinstance(mod, torch.jit.ScriptModule):
            raise TypeError("fp32_cnn must not contain TorchScript submodules.")
        if "quantized" in type(mod).__module__:
            raise ValueError("fp32_cnn must be an FP32 module without quantized submodules.")
    for param in fp32_cnn.parameters():
        if not param.is_floating_point():
            raise ValueError("fp32_cnn must contain only floating-point parameters.")
    layer_getter = getattr(fp32_cnn, "target_gradcam_layer", None)
    if not callable(layer_getter):
        raise TypeError("fp32_cnn must expose a callable target_gradcam_layer() method.")
    target_layer = layer_getter()
    if not isinstance(target_layer, nn.Module):
        raise TypeError("target_gradcam_layer() must return a torch.nn.Module.")
    if isinstance(target_layer, torch.jit.ScriptModule):
        raise TypeError("target_gradcam_layer() must not return a ScriptModule.")
    if "quantized" in type(target_layer).__module__:
        raise ValueError("target_gradcam_layer() must not return a quantized module.")
    return target_layer


def _validate_inputs(input_tensor: torch.Tensor, target_class: int | None) -> None:
    """Validate input tensor shape/dtype/finiteness and optional target class index."""
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("input_tensor_1x1x28x28 must be a torch.Tensor.")
    if tuple(input_tensor.shape) != _INPUT_SHAPE:
        raise ValueError("input_tensor_1x1x28x28 must have shape (1, 1, 28, 28).")
    if not input_tensor.is_floating_point():
        raise TypeError("input_tensor_1x1x28x28 must be a floating-point tensor.")
    if not bool(torch.isfinite(input_tensor).all().item()):
        raise ValueError("input_tensor_1x1x28x28 must contain only finite values.")
    if target_class is not None:
        if isinstance(target_class, bool) or not isinstance(target_class, int):
            raise TypeError("target_class must be None or a non-negative integer.")
        if target_class < 0:
            raise ValueError("target_class must be >= 0.")


def _resolve_model_device(fp32_cnn: nn.Module, fallback: torch.device) -> torch.device:
    """Return the device of the model's parameters or buffers, or fallback."""
    for param in fp32_cnn.parameters():
        return param.device
    for buf in fp32_cnn.buffers():
        return buf.device
    return fallback


def _select_target_class(logits: torch.Tensor, target_class: int | None) -> int:
    """Validate 2D logits and resolve the class index to explain."""
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != _LOGITS_NDIM
        or logits.shape[0] != 1
        or logits.shape[1] <= 0
    ):
        raise ValueError("fp32_cnn must return 2D logits with shape (1, num_classes).")
    if target_class is None:
        return int(torch.argmax(logits[0]).item())
    if target_class >= logits.shape[-1]:
        raise ValueError(
            f"target_class {target_class} is out of range for {logits.shape[-1]} classes."
        )
    return target_class


def _extract_4d_acts_and_grads(
    activations: list[torch.Tensor],
    gradients: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate captured activations and gradients and return detached 4D tensors."""
    if not activations or not gradients:
        raise ValueError("Target Grad-CAM layer did not produce activations and gradients.")
    raw_acts = activations[0]
    raw_grads = gradients[0]
    if not isinstance(raw_acts, torch.Tensor) or not isinstance(raw_grads, torch.Tensor):
        raise TypeError("Target Grad-CAM layer must produce tensor activations and gradients.")
    acts = raw_acts.detach()
    grads = raw_grads.detach()
    if (
        acts.ndim != _FEATURE_MAP_NDIM
        or grads.ndim != _FEATURE_MAP_NDIM
        or acts.shape != grads.shape
        or acts.shape[0] != 1
        or acts.shape[2] <= 0
        or acts.shape[3] <= 0
    ):
        raise ValueError(
            "Target Grad-CAM layer activations and gradients must share shape (1, C, H, W)."
        )
    return acts, grads


def _capture_activations_and_gradients(
    fp32_cnn: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run forward/autograd pass with local hooks without mutating model or tensor state."""
    model_device = _resolve_model_device(fp32_cnn, input_tensor.device)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations.append(output)

    def bwd_hook(
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        if grad_output and grad_output[0] is not None:
            gradients.append(grad_output[0])

    was_training = fp32_cnn.training
    fwd_handle = bwd_handle = None
    try:
        if was_training:
            fp32_cnn.eval()
        fwd_handle = target_layer.register_forward_hook(fwd_hook)
        bwd_handle = target_layer.register_full_backward_hook(bwd_hook)
        with torch.inference_mode(False), torch.enable_grad():
            x = input_tensor.detach().to(device=model_device, dtype=torch.float32)
            x = x.clone().requires_grad_(True)
            logits = fp32_cnn(x)
            cls_idx = _select_target_class(logits, target_class)
            if not logits[0, cls_idx].requires_grad:
                raise ValueError("Selected class logit is not differentiable with respect to input.")
            torch.autograd.grad(logits[0, cls_idx], x)
    finally:
        if fwd_handle is not None:
            fwd_handle.remove()
        if bwd_handle is not None:
            bwd_handle.remove()
        if was_training:
            fp32_cnn.train(True)
    return _extract_4d_acts_and_grads(activations, gradients)


def _compute_cam_28x28(acts: torch.Tensor, grads: torch.Tensor) -> np.ndarray | None:
    """Compute ReLU-masked Grad-CAM upsampled to 28x28, or None when flat/non-finite."""
    acts_f = acts.to(dtype=torch.float32)
    grads_f = grads.to(dtype=torch.float32)
    weights = grads_f.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts_f).sum(dim=1, keepdim=True))
    cam_max = float(cam.max().item())
    if not bool(torch.isfinite(cam).all().item()) or not math.isfinite(cam_max) or cam_max <= _CAM_EPS:
        return None
    cam_norm = cam / cam_max
    upsampled = F.interpolate(
        cam_norm,
        size=_OUTPUT_SIZE,
        mode="bilinear",
        align_corners=False,
    )
    return upsampled[0, 0].cpu().numpy().clip(0.0, 1.0).astype(np.float32, copy=False)


def _normalize_grayscale_28x28(input_tensor: torch.Tensor) -> np.ndarray:
    """Convert the (1, 1, 28, 28) input tensor into a finite [0.0, 1.0] 28x28 array."""
    gray = input_tensor.detach()[0, 0].to(device="cpu", dtype=torch.float32).numpy()
    g_min = float(np.min(gray))
    g_max = float(np.max(gray))
    span = g_max - g_min
    if not math.isfinite(span) or span <= _GRAY_SPAN_EPS:
        return np.clip(gray, 0.0, 1.0).astype(np.float32, copy=True)
    if g_min >= 0.0 and g_max <= 1.0:
        return np.clip(gray, 0.0, 1.0).astype(np.float32, copy=True)
    return np.clip((gray - g_min) / span, 0.0, 1.0).astype(np.float32, copy=False)


def _encode_overlay_png_b64(gray: np.ndarray, cam_28: np.ndarray | None) -> str:
    """Blend grayscale input with jet heatmap (or clean grayscale RGB) and encode as PNG."""
    if cam_28 is None:
        blended = np.repeat(gray[..., None], _RGB_CHANNELS, axis=2)
    else:
        heat_rgb = np.asarray(
            matplotlib.colormaps[_JET_COLORMAP_NAME](cam_28)[..., :_RGB_CHANNELS],
            dtype=np.float32,
        )
        blended = np.clip(
            _GRAY_BLEND_WEIGHT * gray[..., None] + _HEAT_BLEND_WEIGHT * heat_rgb,
            0.0,
            1.0,
        )
    rgb_u8 = np.rint(np.clip(blended, 0.0, 1.0) * _UINT8_MAX).astype(np.uint8)
    img = Image.fromarray(rgb_u8)
    buf = io.BytesIO()
    try:
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    finally:
        img.close()
        buf.close()


def generate_gradcam_b64(
    fp32_cnn: nn.Module,
    input_tensor_1x1x28x28: torch.Tensor,
    target_class: int | None = None,
) -> str:
    """Compute an FP32 Grad-CAM heatmap overlay and return a base64-encoded RGB PNG."""
    target_layer = _validate_fp32_cnn(fp32_cnn)
    _validate_inputs(input_tensor_1x1x28x28, target_class)
    acts, grads = _capture_activations_and_gradients(
        fp32_cnn,
        target_layer,
        input_tensor_1x1x28x28,
        target_class,
    )
    cam_28 = _compute_cam_28x28(acts, grads)
    gray = _normalize_grayscale_28x28(input_tensor_1x1x28x28)
    return _encode_overlay_png_b64(gray, cam_28)

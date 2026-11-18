"""Spec-driven tests for vision models, Grad-CAM, quantization, and training artifacts."""

from __future__ import annotations

import base64
import errno
import io
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.ao.nn.quantized.dynamic import Linear as DynamicQuantizedLinear
from torch.ao.quantization import DeQuantStub, QuantStub
from torchvision import transforms

from data.preprocess import (
    CANONICAL_36_CHARS,
    EMNIST_BALANCED_CHARS,
    fix_emnist_orientation,
)
from models.char_models import (
    INPUT_DIM,
    NUM_CLASSES,
    CharCNN,
    DenseMLP,
    load_char_cnn_fp32,
    load_dense_mlp,
    save_fp32_checkpoint,
)
from models.gradcam import generate_gradcam_b64
from models.quantization import (
    benchmark_cpu_latency,
    get_model_disk_size_mb,
    get_quantization_engine,
    load_quantized_bert_dynamic,
    load_quantized_cnn,
    quantize_bert_dynamic,
    quantize_cnn_static,
    save_quantized_cnn,
    save_quantized_state_dict,
)
from scripts.train_vision import (
    apply_random_affine_batch,
    build_emnist_transform,
    compute_stratified_split,
    evaluate_vision_model,
    extract_oriented_tensors,
    load_emnist_datasets,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PAIRED_LOWERCASE = ("a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t")
WEIGHTS_DIR = Path("weights")
RESULTS_DIR = Path("results")


class _TinyTransformerBlock(nn.Module):
    """Lightweight offline Transformer-like module for dynamic INT8 quantization tests."""

    def __init__(self, vocab_size: int = 32, hidden_dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.embed(input_ids))
        return self.fc2(self.act(self.fc1(h)))


def _decode_png_b64(b64_str: str) -> tuple[bytes, Image.Image, np.ndarray]:
    assert isinstance(b64_str, str) and len(b64_str) > 0
    raw = base64.b64decode(b64_str, validate=True)
    assert raw.startswith(PNG_MAGIC)
    img = Image.open(io.BytesIO(raw))
    img.load()
    arr = np.asarray(img)
    return raw, img, arr


def _make_corner_blobs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    top_left = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    top_left[0, 0, 2:10, 2:10] = 1.0
    bottom_right = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    bottom_right[0, 0, 18:26, 18:26] = 1.0
    both = torch.clamp(top_left + bottom_right, 0.0, 1.0)
    return top_left, bottom_right, both


def _make_upright_f_image() -> np.ndarray:
    arr = np.zeros((28, 28), dtype=np.float32)
    arr[4:24, 6:9] = 1.0
    arr[4:7, 6:19] = 1.0
    arr[12:15, 6:15] = 1.0
    return arr


def _assert_friendly_error(exc: BaseException) -> None:
    msg = str(exc)
    assert msg and "\n" not in msg and "Traceback" not in msg


def _count_module_hooks(model: nn.Module) -> int:
    total = 0
    for mod in model.modules():
        total += len(mod._forward_hooks)
        total += len(mod._backward_hooks)
        total += len(mod._forward_pre_hooks)
        total += len(mod._backward_pre_hooks)
    return total


def _load_quantized_weights_only(path: Path) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"TypedStorage is deprecated.*",
            category=UserWarning,
        )
        return torch.load(path, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# 1. DenseMLP
# ---------------------------------------------------------------------------


def test_dense_mlp_architecture_dimensions_and_dropout() -> None:
    assert NUM_CLASSES == 47 and INPUT_DIM == 784
    mlp = DenseMLP()
    linear_dims = [
        (m.in_features, m.out_features) for m in mlp.modules() if isinstance(m, nn.Linear)
    ]
    assert linear_dims == [(784, 512), (512, 256), (256, 47)]

    relus = [m for m in mlp.modules() if isinstance(m, nn.ReLU)]
    dropouts = [m for m in mlp.modules() if isinstance(m, nn.Dropout)]
    assert len(relus) == 2
    assert len(dropouts) == 2
    assert all(math.isclose(d.p, 0.2, rel_tol=0.0, abs_tol=1e-7) for d in dropouts)

    custom = DenseMLP(num_classes=12, dropout=0.35)
    custom_linears = [
        (m.in_features, m.out_features) for m in custom.modules() if isinstance(m, nn.Linear)
    ]
    assert custom_linears[-1] == (256, 12)
    assert all(
        math.isclose(m.p, 0.35, rel_tol=0.0, abs_tol=1e-7)
        for m in custom.modules()
        if isinstance(m, nn.Dropout)
    )

    for bad_cls in (0, -1, True, "47"):
        with pytest.raises((TypeError, ValueError)):
            DenseMLP(num_classes=bad_cls)  # type: ignore[arg-type]
    for bad_drop in (-0.1, 1.0, 1.5, True, "0.2"):
        with pytest.raises((TypeError, ValueError)):
            DenseMLP(dropout=bad_drop)  # type: ignore[arg-type]


def test_dense_mlp_forward_shapes_nonlinearity_dropout_and_validation() -> None:
    torch.manual_seed(7)
    mlp = DenseMLP()
    x_img = torch.rand((4, 1, 28, 28), dtype=torch.float32)
    x_flat = x_img.view(4, 784)

    mlp.eval()
    out_img = mlp(x_img)
    out_flat = mlp(x_flat)
    assert out_img.shape == (4, 47) and out_img.dtype == torch.float32
    assert torch.isfinite(out_img).all()
    assert torch.allclose(out_img, out_flat, atol=1e-6)

    # Purely linear network would satisfy f(2x) == 2*f(x) - f(0); ReLU must break it.
    zero_out = mlp(torch.zeros_like(x_flat))
    scaled_out = mlp(2.0 * x_flat)
    assert not torch.allclose(scaled_out, 2.0 * out_flat - zero_out, atol=1e-4)

    # Dropout must introduce stochasticity in train() mode.
    mlp.train()
    train_a = mlp(x_flat)
    train_b = mlp(x_flat)
    assert not torch.allclose(train_a, train_b, atol=1e-6)

    assert mlp(torch.zeros((0, 784), dtype=torch.float32)).shape == (0, 47)
    for bad_shape in ((784,), (1, 28, 28), (2, 2, 28, 28), (2, 1, 27, 28), (2, 1, 28, 29), (2, 783)):
        with pytest.raises((TypeError, ValueError)):
            mlp(torch.zeros(bad_shape, dtype=torch.float32))
    for bad_obj in (None, "bad", np.zeros((2, 784), dtype=np.float32)):
        with pytest.raises((TypeError, ValueError)):
            mlp(bad_obj)  # type: ignore[arg-type]
    for bad_dtype in (torch.int64, torch.bool, torch.complex64):
        with pytest.raises((TypeError, ValueError)):
            mlp(torch.zeros((2, 784), dtype=bad_dtype))


def test_dense_mlp_sgd_backprop_training_reduces_loss_and_updates_params() -> None:
    torch.manual_seed(11)
    mlp = DenseMLP()
    mlp.train()

    inputs = torch.zeros((16, 1, 28, 28), dtype=torch.float32)
    targets = torch.tensor([0, 10, 25, 46] * 4, dtype=torch.long)
    for idx, cls_id in enumerate(targets.tolist()):
        row = (cls_id % 20) + 2
        inputs[idx, 0, row : row + 4, 4:24] = 1.0

    initial_params = [p.detach().clone() for p in mlp.parameters()]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=0.05, momentum=0.9)

    mlp.eval()
    with torch.no_grad():
        initial_loss = float(criterion(mlp(inputs), targets))

    mlp.train()
    for _ in range(25):
        optimizer.zero_grad()
        loss = criterion(mlp(inputs), targets)
        loss.backward()
        optimizer.step()

    mlp.eval()
    with torch.no_grad():
        final_loss = float(criterion(mlp(inputs), targets))

    assert final_loss < initial_loss * 0.5
    for before, after in zip(initial_params, mlp.parameters(), strict=True):
        assert not torch.equal(before, after.detach())


def test_dense_mlp_checkpoint_roundtrip_raw_state_dict_and_errors(tmp_path: Path) -> None:
    torch.manual_seed(3)
    mlp = DenseMLP()
    mlp.train()
    x = torch.rand((3, 1, 28, 28), dtype=torch.float32)

    ckpt_path = tmp_path / "nested" / "dense_mlp.pt"
    saved_path = save_fp32_checkpoint(mlp, ckpt_path, metadata={"epoch": 2, "tag": "unit"})
    assert saved_path == ckpt_path and ckpt_path.is_file()

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict) and payload["num_classes"] == 47

    mlp.eval()
    reloaded = load_dense_mlp(ckpt_path, device="cpu")
    assert isinstance(reloaded, DenseMLP) and not reloaded.training
    assert torch.allclose(mlp(x), reloaded(x), atol=1e-6)

    raw_path = tmp_path / "dense_raw.pt"
    torch.save(mlp.state_dict(), raw_path)
    reloaded_raw = load_dense_mlp(raw_path)
    assert not reloaded_raw.training and torch.allclose(mlp(x), reloaded_raw(x), atol=1e-6)

    custom = DenseMLP(num_classes=15)
    custom.eval()
    custom_path = tmp_path / "dense_custom.pt"
    save_fp32_checkpoint(custom, custom_path)
    reloaded_custom = load_dense_mlp(custom_path)
    assert reloaded_custom(x).shape == (3, 15)
    assert torch.allclose(custom(x), reloaded_custom(x), atol=1e-6)

    with pytest.raises(FileNotFoundError) as missing_exc:
        load_dense_mlp(tmp_path / "does_not_exist.pt")
    assert missing_exc.value.errno == errno.ENOENT
    _assert_friendly_error(missing_exc.value)

    with pytest.raises(IsADirectoryError) as dir_exc:
        load_dense_mlp(tmp_path)
    _assert_friendly_error(dir_exc.value)

    corrupt_path = tmp_path / "corrupt.pt"
    corrupt_path.write_bytes(b"not a valid torch archive")
    with pytest.raises(ValueError) as corrupt_exc:
        load_dense_mlp(corrupt_path)
    _assert_friendly_error(corrupt_exc.value)

    bad_payload = tmp_path / "bad_payload.pt"
    torch.save({"unexpected": 1}, bad_payload)
    with pytest.raises(ValueError):
        load_dense_mlp(bad_payload)

    cnn_path = tmp_path / "cnn_wrong_arch.pt"
    save_fp32_checkpoint(CharCNN(), cnn_path)
    with pytest.raises(ValueError):
        load_dense_mlp(cnn_path)

    with pytest.raises(IsADirectoryError):
        save_fp32_checkpoint(mlp, tmp_path)


# ---------------------------------------------------------------------------
# 2. CharCNN
# ---------------------------------------------------------------------------


def test_char_cnn_architecture_pools_blocks_and_dropout() -> None:
    cnn = CharCNN()
    assert any(isinstance(m, QuantStub) for m in cnn.modules())
    assert any(isinstance(m, DeQuantStub) for m in cnn.modules())

    pools = [m for m in cnn.modules() if isinstance(m, nn.MaxPool2d)]
    assert len(pools) == 2

    b1_convs = [m for m in cnn.block1.modules() if isinstance(m, nn.Conv2d)]
    b1_bns = [m for m in cnn.block1.modules() if isinstance(m, nn.BatchNorm2d)]
    b1_pools = [m for m in cnn.block1.modules() if isinstance(m, nn.MaxPool2d)]
    assert len(b1_convs) == 1 and b1_convs[0].in_channels == 1 and b1_convs[0].out_channels == 32
    assert b1_convs[0].kernel_size == (3, 3) and b1_convs[0].padding == (1, 1)
    assert b1_convs[0].bias is None
    assert len(b1_bns) == 1 and b1_bns[0].num_features == 32 and len(b1_pools) == 1

    b2_convs = [m for m in cnn.block2.modules() if isinstance(m, nn.Conv2d)]
    b2_bns = [m for m in cnn.block2.modules() if isinstance(m, nn.BatchNorm2d)]
    b2_pools = [m for m in cnn.block2.modules() if isinstance(m, nn.MaxPool2d)]
    assert len(b2_convs) == 1 and b2_convs[0].in_channels == 32 and b2_convs[0].out_channels == 64
    assert b2_convs[0].kernel_size == (3, 3) and b2_convs[0].padding == (1, 1)
    assert b2_convs[0].bias is None
    assert len(b2_bns) == 1 and b2_bns[0].num_features == 64 and len(b2_pools) == 1

    b3_convs = [m for m in cnn.block3.modules() if isinstance(m, nn.Conv2d)]
    b3_bns = [m for m in cnn.block3.modules() if isinstance(m, nn.BatchNorm2d)]
    b3_pools = [m for m in cnn.block3.modules() if isinstance(m, nn.MaxPool2d)]
    assert len(b3_convs) == 1 and b3_convs[0].in_channels == 64 and b3_convs[0].out_channels == 128
    assert b3_convs[0].kernel_size == (3, 3) and b3_convs[0].padding == (1, 1)
    assert b3_convs[0].bias is None
    assert len(b3_bns) == 1 and b3_bns[0].num_features == 128 and len(b3_pools) == 0

    clf_drops = [m for m in cnn.classifier.modules() if isinstance(m, nn.Dropout)]
    clf_lins = [m for m in cnn.classifier.modules() if isinstance(m, nn.Linear)]
    assert [d.p for d in clf_drops] == pytest.approx([0.3, 0.2])
    assert [(ln.in_features, ln.out_features) for ln in clf_lins] == [
        (128 * 7 * 7, 256),
        (256, 47),
    ]

    custom_cnn = CharCNN(num_classes=19)
    custom_lins = [m for m in custom_cnn.classifier.modules() if isinstance(m, nn.Linear)]
    assert (custom_lins[-1].in_features, custom_lins[-1].out_features) == (256, 19)
    with pytest.raises((TypeError, ValueError)):
        CharCNN(num_classes=0)


def test_char_cnn_intermediate_7x7_feature_map_and_input_validation() -> None:
    torch.manual_seed(5)
    cnn = CharCNN()
    cnn.eval()

    x = torch.rand((3, 1, 28, 28), dtype=torch.float32)
    captured_target: list[torch.Tensor] = []
    captured_block3: list[torch.Tensor] = []

    h_target = cnn.target_gradcam_layer().register_forward_hook(
        lambda _mod, _inp, out: captured_target.append(out.detach())
    )
    h_block3 = cnn.block3.register_forward_hook(
        lambda _mod, _inp, out: captured_block3.append(out.detach())
    )
    try:
        logits = cnn(x)
    finally:
        h_target.remove()
        h_block3.remove()

    assert logits.shape == (3, 47) and torch.isfinite(logits).all()
    assert len(captured_target) == 1 and captured_target[0].shape == (3, 128, 7, 7)
    assert len(captured_block3) == 1 and captured_block3[0].shape == (3, 128, 7, 7)
    assert _count_module_hooks(cnn) == 0

    cnn.train()
    assert not torch.allclose(cnn(x), cnn(x), atol=1e-6)

    assert cnn(torch.zeros((0, 1, 28, 28), dtype=torch.float32)).shape == (0, 47)
    for bad_shape in ((3, 784), (1, 28, 28), (28, 28), (2, 3, 28, 28), (2, 1, 27, 28), (2, 1, 28, 29)):
        with pytest.raises((TypeError, ValueError)):
            cnn(torch.zeros(bad_shape, dtype=torch.float32))
    for bad_dtype in (torch.int64, torch.bool, torch.complex64):
        with pytest.raises((TypeError, ValueError)):
            cnn(torch.zeros((2, 1, 28, 28), dtype=bad_dtype))
    with pytest.raises((TypeError, ValueError)):
        cnn("not a tensor")  # type: ignore[arg-type]


def test_char_cnn_fuse_model_eval_parity_idempotency_and_gradcam_layer() -> None:
    torch.manual_seed(13)
    cnn = CharCNN()
    cnn.train()
    # Populate non-default BatchNorm running statistics before fusion.
    _ = cnn(torch.rand((6, 1, 28, 28), dtype=torch.float32) * 2.0 + 0.5)

    x = torch.rand((2, 1, 28, 28), dtype=torch.float32)
    cnn.eval()
    before_logits = cnn(x).detach()
    pre_layer = cnn.target_gradcam_layer()
    assert isinstance(pre_layer, nn.Module)

    cnn.train()
    cnn.fuse_model()
    assert not cnn.training
    assert not any(isinstance(m, nn.BatchNorm2d) for m in cnn.modules())

    after_logits = cnn(x).detach()
    assert torch.allclose(after_logits, before_logits, atol=1e-4)

    # Idempotency check when called a second time.
    cnn.fuse_model()
    assert not cnn.training
    assert torch.allclose(cnn(x).detach(), before_logits, atol=1e-4)

    post_layer = cnn.target_gradcam_layer()
    assert isinstance(post_layer, nn.Module)
    assert not cnn.training

    captured: list[torch.Tensor] = []
    handle = post_layer.register_forward_hook(
        lambda _mod, _inp, out: captured.append(out.detach())
    )
    try:
        _ = cnn(x)
    finally:
        handle.remove()
    assert len(captured) == 1 and captured[0].shape == (2, 128, 7, 7)


def test_char_cnn_checkpoint_roundtrip_and_errors(tmp_path: Path) -> None:
    torch.manual_seed(17)
    cnn = CharCNN()
    cnn.eval()
    x = torch.rand((2, 1, 28, 28), dtype=torch.float32)

    ckpt_path = tmp_path / "char_cnn_fp32.pt"
    save_fp32_checkpoint(cnn, ckpt_path, metadata={"arch": "char_cnn"})
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert payload["num_classes"] == 47

    reloaded = load_char_cnn_fp32(ckpt_path)
    assert isinstance(reloaded, CharCNN) and not reloaded.training
    assert torch.allclose(cnn(x), reloaded(x), atol=1e-6)

    raw_path = tmp_path / "cnn_raw.pt"
    torch.save(cnn.state_dict(), raw_path)
    assert torch.allclose(cnn(x), load_char_cnn_fp32(raw_path)(x), atol=1e-6)

    with pytest.raises(FileNotFoundError) as missing_exc:
        load_char_cnn_fp32(tmp_path / "missing_cnn.pt")
    assert missing_exc.value.errno == errno.ENOENT
    _assert_friendly_error(missing_exc.value)

    with pytest.raises(IsADirectoryError):
        load_char_cnn_fp32(tmp_path)

    bad_path = tmp_path / "bad_cnn.pt"
    bad_path.write_bytes(b"corrupt checkpoint")
    with pytest.raises(ValueError):
        load_char_cnn_fp32(bad_path)

    mlp_path = tmp_path / "mlp_for_cnn.pt"
    save_fp32_checkpoint(DenseMLP(), mlp_path)
    with pytest.raises(ValueError):
        load_char_cnn_fp32(mlp_path)


# ---------------------------------------------------------------------------
# 3. Grad-CAM
# ---------------------------------------------------------------------------


def test_gradcam_valid_png_and_works_under_no_grad_and_inference_mode() -> None:
    torch.manual_seed(21)
    cnn = CharCNN().eval()
    x = torch.rand((1, 1, 28, 28), dtype=torch.float32)

    b64_normal = generate_gradcam_b64(cnn, x)
    _, img_normal, arr_normal = _decode_png_b64(b64_normal)
    assert img_normal.size == (28, 28) and img_normal.mode == "RGB"
    assert arr_normal.shape == (28, 28, 3) and np.isfinite(arr_normal).all()

    with torch.no_grad():
        b64_nograd = generate_gradcam_b64(cnn, x)
    with torch.inference_mode():
        b64_infer = generate_gradcam_b64(cnn, x)

    _, _, arr_nograd = _decode_png_b64(b64_nograd)
    _, _, arr_infer = _decode_png_b64(b64_infer)
    assert np.array_equal(arr_normal, arr_nograd)
    assert np.array_equal(arr_normal, arr_infer)

    # Explicit target classes including boundary 0 and 46, and after fuse_model().
    for cls_idx in (0, 46):
        _, img_cls, arr_cls = _decode_png_b64(generate_gradcam_b64(cnn, x, target_class=cls_idx))
        assert img_cls.size == (28, 28) and arr_cls.shape == (28, 28, 3)

    cnn.fuse_model()
    _, img_fused, arr_fused = _decode_png_b64(generate_gradcam_b64(cnn, x, target_class=0))
    assert img_fused.size == (28, 28) and arr_fused.shape == (28, 28, 3)


def test_gradcam_zero_input_and_zero_cam_nan_guard() -> None:
    torch.manual_seed(23)
    cnn = CharCNN().eval()
    _, img_zero_in, arr_zero_in = _decode_png_b64(
        generate_gradcam_b64(cnn, torch.zeros((1, 1, 28, 28), dtype=torch.float32))
    )
    assert img_zero_in.size == (28, 28) and img_zero_in.mode == "RGB"
    assert np.isfinite(arr_zero_in).all()

    # Zero-activation & zero-gradient model triggers cam.max() <= 1e-8 guard.
    zero_cnn = CharCNN().eval()
    with torch.no_grad():
        for param in zero_cnn.parameters():
            param.zero_()
    _, _, x_both = _make_corner_blobs()
    _, img_zero_cam, arr_zero_cam = _decode_png_b64(
        generate_gradcam_b64(zero_cnn, x_both, target_class=0)
    )
    assert img_zero_cam.size == (28, 28) and img_zero_cam.mode == "RGB"
    assert np.isfinite(arr_zero_cam).all()
    # Clean grayscale RGB fallback has R == G == B across all pixels.
    assert np.array_equal(arr_zero_cam[:, :, 0], arr_zero_cam[:, :, 1])
    assert np.array_equal(arr_zero_cam[:, :, 1], arr_zero_cam[:, :, 2])
    assert int(arr_zero_cam.max()) > int(arr_zero_cam.min())


def test_gradcam_no_hook_leaks_and_no_model_or_grad_mutation() -> None:
    torch.manual_seed(29)
    for mode_train in (True, False):
        cnn = CharCNN()
        cnn.train(mode_train)
        x = torch.rand((1, 1, 28, 28), dtype=torch.float32, requires_grad=True)

        dict_keys_before = set(cnn.__dict__.keys())
        training_before = cnn.training
        params_before = [p.detach().clone() for p in cnn.parameters()]
        assert all(p.grad is None for p in cnn.parameters())

        for _ in range(5):
            _decode_png_b64(generate_gradcam_b64(cnn, x))

        with pytest.raises((ValueError, IndexError)):
            generate_gradcam_b64(cnn, x, target_class=999)

        assert _count_module_hooks(cnn) == 0
        assert cnn.training == training_before
        assert set(cnn.__dict__.keys()) == dict_keys_before
        assert all(p.grad is None for p in cnn.parameters())
        assert x.grad is None
        for before, after in zip(params_before, cnn.parameters(), strict=True):
            assert torch.equal(before, after.detach())


def test_gradcam_spatial_responsiveness() -> None:
    torch.manual_seed(31)
    cnn = CharCNN()
    top_left, bottom_right, both = _make_corner_blobs()

    # Briefly fit 2 opposite corners to classes 0 and 1 so spatial grads diverge strongly.
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    batch_x = torch.cat([top_left, bottom_right], dim=0)
    batch_y = torch.tensor([0, 1], dtype=torch.long)
    cnn.train()
    for _ in range(8):
        optimizer.zero_grad()
        loss = criterion(cnn(batch_x), batch_y)
        loss.backward()
        optimizer.step()
    cnn.eval()

    _, _, arr_c0 = _decode_png_b64(generate_gradcam_b64(cnn, both, target_class=0))
    _, _, arr_c1 = _decode_png_b64(generate_gradcam_b64(cnn, both, target_class=1))
    _, _, arr_tl = _decode_png_b64(generate_gradcam_b64(cnn, top_left, target_class=0))
    _, _, arr_br = _decode_png_b64(generate_gradcam_b64(cnn, bottom_right, target_class=1))

    # Same composite input with different classes must produce distinct heatmaps.
    assert float(np.abs(arr_c0.astype(float) - arr_c1.astype(float)).mean()) > 1.0
    assert float(np.abs(arr_tl.astype(float) - arr_br.astype(float)).mean()) > 1.0

    # Jet colormap blending produces non-grayscale RGB pixels.
    chroma_c0 = (arr_c0.max(axis=-1) - arr_c0.min(axis=-1)).max()
    chroma_c1 = (arr_c1.max(axis=-1) - arr_c1.min(axis=-1)).max()
    assert int(chroma_c0) > 10 and int(chroma_c1) > 10


def test_gradcam_rejects_int8_scripted_mlp_and_bad_inputs() -> None:
    torch.manual_seed(37)
    cnn = CharCNN().eval()
    x = torch.rand((1, 1, 28, 28), dtype=torch.float32)

    int8_cnn = quantize_cnn_static(cnn, [torch.rand((4, 1, 28, 28), dtype=torch.float32)])
    with pytest.raises((TypeError, ValueError)):
        generate_gradcam_b64(int8_cnn, x)

    scripted = torch.jit.script(CharCNN().eval())
    with pytest.raises((TypeError, ValueError)):
        generate_gradcam_b64(scripted, x)  # type: ignore[arg-type]

    with pytest.raises((TypeError, ValueError)):
        generate_gradcam_b64(DenseMLP().eval(), x)

    for bad_shape in ((2, 1, 28, 28), (1, 28, 28), (1, 3, 28, 28), (1, 1, 27, 28), (1, 784)):
        with pytest.raises((TypeError, ValueError)):
            generate_gradcam_b64(cnn, torch.zeros(bad_shape, dtype=torch.float32))

    for bad_dtype in (torch.int64, torch.bool):
        with pytest.raises((TypeError, ValueError)):
            generate_gradcam_b64(cnn, torch.zeros((1, 1, 28, 28), dtype=bad_dtype))

    nan_x = x.clone()
    nan_x[0, 0, 0, 0] = float("nan")
    inf_x = x.clone()
    inf_x[0, 0, 0, 0] = float("inf")
    for bad_val in (nan_x, inf_x, "bad"):
        with pytest.raises((TypeError, ValueError)):
            generate_gradcam_b64(cnn, bad_val)  # type: ignore[arg-type]

    for bad_cls in (True, False, -1, 47, 999, "0", 1.5):
        with pytest.raises((TypeError, ValueError, IndexError)):
            generate_gradcam_b64(cnn, x, target_class=bad_cls)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. Quantization & CPU Benchmarks
# ---------------------------------------------------------------------------


def test_get_quantization_engine_dynamic_selection_and_validation() -> None:
    supported = list(torch.backends.quantized.supported_engines)
    engine = get_quantization_engine()
    assert isinstance(engine, str) and engine in supported
    if "onednn" in supported:
        assert engine == "onednn"
    assert get_quantization_engine(engine) == engine

    with pytest.raises(ValueError):
        get_quantization_engine("unknown_quant_engine_xyz")
    with pytest.raises((TypeError, ValueError)):
        get_quantization_engine(123)  # type: ignore[arg-type]


def test_quantize_cnn_static_non_mutating_and_save_load_parity(tmp_path: Path) -> None:
    torch.manual_seed(41)
    fp32_cnn = CharCNN()
    fp32_cnn.train()
    orig_layer_type = type(fp32_cnn.target_gradcam_layer())
    orig_params = [p.detach().clone() for p in fp32_cnn.parameters()]

    calib_batches = [
        (torch.rand((4, 1, 28, 28), dtype=torch.float32), torch.zeros(4, dtype=torch.long))
        for _ in range(3)
    ]
    int8_cnn = quantize_cnn_static(fp32_cnn, calib_batches)

    # Original FP32 CNN must remain unmutated in its original state.
    assert fp32_cnn.training is True
    assert type(fp32_cnn.target_gradcam_layer()) is orig_layer_type
    assert not any(isinstance(m, nn.Identity) for m in fp32_cnn.modules())
    for before, after in zip(orig_params, fp32_cnn.parameters(), strict=True):
        assert torch.equal(before, after.detach())

    assert not int8_cnn.training
    assert any("quantized" in type(m).__module__ for m in int8_cnn.modules())

    x_test = torch.rand((2, 1, 28, 28), dtype=torch.float32)
    out_int8 = int8_cnn(x_test)
    assert out_int8.shape == (2, 47) and torch.isfinite(out_int8).all()

    # Raw tensor iterable (not (x, y) tuples) also calibrates cleanly.
    int8_raw = quantize_cnn_static(fp32_cnn, [torch.rand((4, 1, 28, 28), dtype=torch.float32)])
    assert int8_raw(x_test).shape == (2, 47)

    q_path = tmp_path / "char_cnn_int8.pt"
    saved = save_quantized_cnn(int8_cnn, q_path)
    assert saved == q_path and q_path.is_file()
    raw_payload = _load_quantized_weights_only(q_path)
    assert isinstance(raw_payload, dict)

    reloaded = load_quantized_cnn(q_path)
    assert not reloaded.training
    assert torch.allclose(out_int8, reloaded(x_test), atol=1e-6)

    fp_path = tmp_path / "char_cnn_fp32.pt"
    save_fp32_checkpoint(fp32_cnn, fp_path)
    assert get_model_disk_size_mb(fp_path) / get_model_disk_size_mb(q_path) >= 2.0

    with pytest.raises(ValueError):
        quantize_cnn_static(fp32_cnn, [])
    with pytest.raises((TypeError, ValueError)):
        quantize_cnn_static(int8_cnn, calib_batches)
    with pytest.raises((TypeError, ValueError)):
        quantize_cnn_static(torch.jit.script(CharCNN().eval()), calib_batches)  # type: ignore[arg-type]

    with pytest.raises(FileNotFoundError) as missing_exc:
        load_quantized_cnn(tmp_path / "missing_int8.pt")
    assert missing_exc.value.errno == errno.ENOENT
    _assert_friendly_error(missing_exc.value)

    with pytest.raises(IsADirectoryError):
        load_quantized_cnn(tmp_path)

    corrupt = tmp_path / "corrupt_int8.pt"
    corrupt.write_bytes(b"corrupt int8 file")
    with pytest.raises(ValueError):
        load_quantized_cnn(corrupt)


def test_quantize_bert_dynamic_save_load_and_no_mutation(tmp_path: Path) -> None:
    torch.manual_seed(43)
    fp32_tiny = _TinyTransformerBlock()
    fp32_tiny.train()
    orig_linear_count = sum(1 for m in fp32_tiny.modules() if isinstance(m, nn.Linear))
    orig_params = [p.detach().clone() for p in fp32_tiny.parameters()]

    int8_tiny = quantize_bert_dynamic(fp32_tiny)

    assert fp32_tiny.training is True
    assert sum(1 for m in fp32_tiny.modules() if isinstance(m, nn.Linear)) == orig_linear_count
    assert not any(isinstance(m, DynamicQuantizedLinear) for m in fp32_tiny.modules())
    for before, after in zip(orig_params, fp32_tiny.parameters(), strict=True):
        assert torch.equal(before, after.detach())

    assert not int8_tiny.training
    dyn_linears = [m for m in int8_tiny.modules() if isinstance(m, DynamicQuantizedLinear)]
    assert len(dyn_linears) == orig_linear_count

    sample_ids = torch.randint(0, 32, (2, 6), dtype=torch.long)
    out_q = int8_tiny(sample_ids)
    assert out_q.shape == (2, 6, 32) and torch.isfinite(out_q).all()

    q_path = tmp_path / "bert_int8.pt"
    saved = save_quantized_state_dict(int8_tiny, q_path, metadata={"arch": "tiny"})
    assert saved == q_path and q_path.is_file()
    payload = _load_quantized_weights_only(q_path)
    assert isinstance(payload, dict)

    reloaded = load_quantized_bert_dynamic(_TinyTransformerBlock, q_path)
    assert not reloaded.training
    assert any(isinstance(m, DynamicQuantizedLinear) for m in reloaded.modules())
    assert torch.allclose(out_q, reloaded(sample_ids), atol=1e-6)

    with pytest.raises((TypeError, ValueError)):
        quantize_bert_dynamic(int8_tiny)
    with pytest.raises(FileNotFoundError):
        load_quantized_bert_dynamic(_TinyTransformerBlock, tmp_path / "missing_bert.pt")
    with pytest.raises(IsADirectoryError):
        load_quantized_bert_dynamic(_TinyTransformerBlock, tmp_path)
    bad = tmp_path / "corrupt_bert.pt"
    bad.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        load_quantized_bert_dynamic(_TinyTransformerBlock, bad)


def test_get_model_disk_size_mb_and_benchmark_cpu_latency(tmp_path: Path) -> None:
    mlp = DenseMLP().eval()
    ckpt = tmp_path / "size_check.pt"
    save_fp32_checkpoint(mlp, ckpt)

    size_path = get_model_disk_size_mb(ckpt)
    size_str = get_model_disk_size_mb(str(ckpt))
    size_mod = get_model_disk_size_mb(mlp)
    size_dict = get_model_disk_size_mb(mlp.state_dict())
    assert size_path == pytest.approx(size_str, rel=1e-9)
    assert all(s > 0.0 for s in (size_path, size_mod, size_dict))

    with pytest.raises(FileNotFoundError):
        get_model_disk_size_mb(tmp_path / "no_file.pt")
    with pytest.raises(IsADirectoryError):
        get_model_disk_size_mb(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        get_model_disk_size_mb(12345)  # type: ignore[arg-type]

    calls = 0
    seen_inference_mode: list[bool] = []
    cnn = CharCNN().eval()

    def _run(batch: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        seen_inference_mode.append(torch.is_inference_mode_enabled())
        return cnn(batch)

    stats = benchmark_cpu_latency(_run, [torch.zeros((1, 1, 28, 28))], warmup=3, runs=12)
    assert calls == 15
    assert all(seen_inference_mode)
    assert set(stats.keys()) == {"mean_ms", "std_ms", "p50_ms", "p95_ms"}
    assert all(isinstance(v, float) and math.isfinite(v) for v in stats.values())
    assert stats["mean_ms"] > 0.0 and stats["std_ms"] >= 0.0
    assert stats["p50_ms"] > 0.0 and stats["p95_ms"] >= stats["p50_ms"]

    for kwargs in (
        {"warmup": -1, "runs": 5},
        {"warmup": True, "runs": 5},
        {"warmup": 0, "runs": 0},
        {"warmup": 0, "runs": False},
    ):
        with pytest.raises((TypeError, ValueError)):
            benchmark_cpu_latency(_run, [torch.zeros((1, 1, 28, 28))], **kwargs)  # type: ignore[arg-type]

    with pytest.raises((TypeError, ValueError)):
        benchmark_cpu_latency("not_callable", [torch.zeros((1, 1, 28, 28))], warmup=0, runs=2)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        benchmark_cpu_latency(_run, [], warmup=0, runs=2)


# ---------------------------------------------------------------------------
# 5. Training Pipeline & Trained Artifacts
# ---------------------------------------------------------------------------


def test_compute_stratified_split_emnist_112800_counts_and_determinism() -> None:
    targets = np.repeat(np.arange(NUM_CLASSES, dtype=np.int64), 2400)
    assert len(targets) == 112_800

    train_idx, val_idx = compute_stratified_split(targets, train_frac=0.9, seed=42)
    assert len(train_idx) == 101_520 and len(val_idx) == 11_280
    assert np.intersect1d(train_idx, val_idx).size == 0
    assert np.array_equal(np.sort(np.concatenate([train_idx, val_idx])), np.arange(112_800))

    assert np.all(np.bincount(targets[train_idx], minlength=NUM_CLASSES) == 2160)
    assert np.all(np.bincount(targets[val_idx], minlength=NUM_CLASSES) == 240)

    train_2, val_2 = compute_stratified_split(torch.from_numpy(targets), train_frac=0.9, seed=42)
    assert np.array_equal(train_idx, train_2) and np.array_equal(val_idx, val_2)

    train_other, _ = compute_stratified_split(targets, train_frac=0.9, seed=99)
    assert not np.array_equal(train_idx, train_other)

    with pytest.raises(ValueError):
        compute_stratified_split(targets, train_frac=1.0)
    with pytest.raises(ValueError):
        compute_stratified_split(np.array([0, 1], dtype=np.int64))


def test_build_emnist_transform_orientation_and_affine_order() -> None:
    tf_eval = build_emnist_transform(augment=False)
    tf_aug = build_emnist_transform(augment=True)
    assert isinstance(tf_eval, transforms.Compose) and isinstance(tf_aug, transforms.Compose)

    assert not any(isinstance(t, transforms.RandomAffine) for t in tf_eval.transforms)
    affine_indices = [i for i, t in enumerate(tf_aug.transforms) if isinstance(t, transforms.RandomAffine)]
    tensor_indices = [i for i, t in enumerate(tf_aug.transforms) if isinstance(t, transforms.ToTensor)]
    assert len(affine_indices) == 1 and len(tensor_indices) == 1
    affine_idx, tensor_idx = affine_indices[0], tensor_indices[0]
    assert tensor_idx < affine_idx

    affine = tf_aug.transforms[affine_idx]
    assert list(affine.degrees) == pytest.approx([-10.0, 10.0])
    assert tuple(affine.translate) == pytest.approx((0.08, 0.08))
    assert tuple(affine.scale) == pytest.approx((0.92, 1.08))

    # Orientation step must run BEFORE RandomAffine and match fix_emnist_orientation.
    f_upright = _make_upright_f_image()
    f_raw_pil = Image.fromarray((f_upright.T * 255.0).astype(np.uint8))
    expected = torch.from_numpy(f_upright).unsqueeze(0)

    eval_out = tf_eval(f_raw_pil)
    assert isinstance(eval_out, torch.Tensor) and torch.allclose(eval_out, expected, atol=1e-6)

    pre_affine = transforms.Compose(tf_aug.transforms[:affine_idx])
    pre_affine_out = pre_affine(f_raw_pil)
    assert torch.allclose(pre_affine_out, expected, atol=1e-6)


@pytest.mark.skipif(not (Path("data/emnist/EMNIST/raw") / "emnist-balanced-test-labels-idx1-ubyte").is_file(), reason="EMNIST balanced files not downloaded")
def test_extract_oriented_tensors_and_apply_random_affine_batch() -> None:
    _, _, test_ds = load_emnist_datasets()
    subset_idx = np.arange(8, dtype=np.int64)
    x_sub, y_sub = extract_oriented_tensors(test_ds, indices=subset_idx)

    assert x_sub.shape == (8, 1, 28, 28) and x_sub.dtype == torch.float32
    assert y_sub.shape == (8,) and y_sub.dtype == torch.int64
    assert 0.0 <= float(x_sub.min()) and float(x_sub.max()) <= 1.0

    expected_x = torch.stack([test_ds[i][0] for i in range(8)], dim=0)
    expected_y = torch.tensor([test_ds[i][1] for i in range(8)], dtype=torch.int64)
    assert torch.allclose(x_sub, expected_x, atol=1e-6)
    assert torch.equal(y_sub, expected_y)

    # Raw dataset slice matches ToTensor -> fix_emnist_orientation directly.
    raw_first = test_ds.data[0].to(dtype=torch.float32) / 255.0
    assert torch.allclose(x_sub[0, 0], fix_emnist_orientation(raw_first), atol=1e-6)

    orig_x = x_sub.clone()
    gen_a = torch.Generator().manual_seed(123)
    gen_b = torch.Generator().manual_seed(123)
    aug_a = apply_random_affine_batch(x_sub, generator=gen_a)
    aug_b = apply_random_affine_batch(x_sub, generator=gen_b)
    assert aug_a.shape == x_sub.shape and aug_a.dtype == torch.float32
    assert 0.0 <= float(aug_a.min()) and float(aug_a.max()) <= 1.0
    assert torch.equal(x_sub, orig_x)
    assert torch.allclose(aug_a, aug_b, atol=1e-6)
    assert not torch.equal(aug_a, x_sub)

    with pytest.raises((TypeError, ValueError)):
        apply_random_affine_batch(x_sub, degrees=-1.0)


def test_evaluate_vision_model_folds_47_to_36_before_argmax() -> None:
    # Construct logits where raw 47-class argmax and max(P(L), P(l)) fail,
    # while P_36(L) = P_47(L) + P_47(l) succeeds for every paired lowercase class.
    class _FixedLogitsModel(nn.Module):
        def __init__(self, logits: torch.Tensor) -> None:
            super().__init__()
            self._logits = logits

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self._logits[: x.shape[0]]

    n_pairs = len(PAIRED_LOWERCASE)
    logits = torch.full((n_pairs, NUM_CLASSES), -30.0, dtype=torch.float32)
    labels = torch.zeros((n_pairs,), dtype=torch.long)

    distractor_idx = CANONICAL_36_CHARS.index("0")
    for row_idx, lower_ch in enumerate(PAIRED_LOWERCASE):
        upper_idx = EMNIST_BALANCED_CHARS.index(lower_ch.upper())
        lower_idx = EMNIST_BALANCED_CHARS.index(lower_ch)
        logits[row_idx, upper_idx] = math.log(0.28)
        logits[row_idx, lower_idx] = math.log(0.27)
        logits[row_idx, distractor_idx] = math.log(0.45)
        labels[row_idx] = lower_idx if row_idx % 2 == 0 else upper_idx

    metrics = evaluate_vision_model(_FixedLogitsModel(logits), torch.zeros((n_pairs, 1, 28, 28)), labels)
    assert metrics["test_acc_47"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["test_acc_47_pct"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["test_acc_36"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["test_acc_36_pct"] == pytest.approx(100.0, abs=1e-4)


@pytest.mark.skipif(not (Path("weights") / "char_cnn_int8.pt").is_file(), reason="trained vision weights not downloaded (python scripts/download_assets.py)")
def test_trained_vision_artifacts_meet_spec_targets_and_reload() -> None:
    mlp_ckpt = WEIGHTS_DIR / "dense_mlp.pt"
    cnn_fp32_ckpt = WEIGHTS_DIR / "char_cnn_fp32.pt"
    cnn_int8_ckpt = WEIGHTS_DIR / "char_cnn_int8.pt"
    metrics_path = RESULTS_DIR / "vision_metrics.json"

    for artifact in (mlp_ckpt, cnn_fp32_ckpt, cnn_int8_ckpt, metrics_path):
        assert artifact.is_file(), f"Missing trained vision artifact: {artifact}"

    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert report["split_counts"] == {"train": 101520, "val": 11280, "test": 18800}
    assert report["cpu_threads"] == 4
    assert report["quantization_engine"] == get_quantization_engine()

    models_report = report["models"]
    mlp_entry = models_report["dense_mlp_fp32"]
    cnn_fp32_entry = models_report["char_cnn_fp32"]
    cnn_int8_entry = models_report["char_cnn_int8"]

    assert cnn_fp32_entry["test_acc_47_pct"] >= 85.0
    assert cnn_fp32_entry["test_acc_47_pct"] > mlp_entry["test_acc_47_pct"]
    assert cnn_fp32_entry["test_acc_47_pct"] - cnn_int8_entry["test_acc_47_pct"] <= 1.5

    fp32_mb = get_model_disk_size_mb(cnn_fp32_ckpt)
    int8_mb = get_model_disk_size_mb(cnn_int8_ckpt)
    assert fp32_mb / int8_mb >= 2.0
    assert report["comparisons"]["int8_cnn_size_reduction_x"] >= 2.0
    assert report["comparisons"]["int8_cnn_acc_47_drop_pp"] <= 1.5

    for entry in (mlp_entry, cnn_fp32_entry, cnn_int8_entry):
        for batch_key in ("latency_cpu_batch_1", "latency_cpu_batch_8"):
            lat = entry[batch_key]
            assert set(lat.keys()) == {"mean_ms", "std_ms", "p50_ms", "p95_ms"}
            assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in lat.values())
            assert lat["mean_ms"] > 0.0 and lat["std_ms"] >= 0.0
            assert lat["p50_ms"] > 0.0 and lat["p95_ms"] >= lat["p50_ms"]

    mlp_loaded = load_dense_mlp(mlp_ckpt)
    cnn_fp32_loaded = load_char_cnn_fp32(cnn_fp32_ckpt)
    cnn_int8_loaded = load_quantized_cnn(cnn_int8_ckpt)

    sample_batch = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
    sample_batch[:, 0, 6:22, 10:18] = 1.0
    for model in (mlp_loaded, cnn_fp32_loaded, cnn_int8_loaded):
        assert not model.training
        logits = model(sample_batch)
        assert logits.shape == (2, NUM_CLASSES) and torch.isfinite(logits).all()

    _, cam_img, cam_arr = _decode_png_b64(
        generate_gradcam_b64(cnn_fp32_loaded, sample_batch[:1])
    )
    assert cam_img.size == (28, 28) and cam_img.mode == "RGB" and np.isfinite(cam_arr).all()

    with pytest.raises((TypeError, ValueError)):
        generate_gradcam_b64(cnn_int8_loaded, sample_batch[:1])

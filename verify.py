"""One-command verification of the project's claims: python verify.py

Prints PASS / FAIL / SKIP per claim and exits 0 only if no check fails. Finishes in well under five
minutes on a CPU with only requirements.txt installed; checks that need the downloaded weights SKIP
(with the download command) when the weights are absent.

  (a) every reported benchmark number re-derives from results/test_predictions.json (no torch);
  (b) the vision models load and classify the committed sample worksheet boxes;
  (c) the INT8 CNN checkpoint loads under this platform's quantized engine and agrees with FP32;
  (d) the app's input validation returns one-line 400s, without loading any model.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"
SAMPLES_DIR = ROOT / "data" / "sample_worksheets"
VISION_WEIGHTS = ("dense_mlp.pt", "char_cnn_fp32.pt", "char_cnn_int8.pt")
DOWNLOAD_HINT = "weights not present; run: python scripts/download_assets.py"
MIN_CNN_SAMPLE_ACC = 0.75  # measured 0.845 on the 71 committed (deliberately confusable/numeric) sample boxes
MIN_DENSE_SAMPLE_ACC = 0.60  # measured 0.718; chance is 1/36
MIN_INT8_AGREEMENT = 0.95  # measured 0.986 (onednn, Windows x86-64)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


def check_recompute() -> CheckResult:
    """(a) Re-derive every reported number from the committed per-item predictions."""
    from scripts.recompute_metrics import compare

    predictions = json.loads((RESULTS_DIR / "test_predictions.json").read_text(encoding="utf-8"))
    results = json.loads((RESULTS_DIR / "benchmark_results.json").read_text(encoding="utf-8"))
    checks = compare(predictions, results)
    failed = [label for label, *_, ok in checks if not ok]
    detail = f"{len(checks) - len(failed)}/{len(checks)} reported numbers re-derive within 0.01" + (f"; mismatched: {', '.join(failed)}" if failed else "")
    return CheckResult("(a) benchmark numbers recompute from per-item predictions", "FAIL" if failed else "PASS", detail)


def _sample_boxes() -> tuple[Any, list[str]]:
    import torch
    from PIL import Image

    from data.preprocess import preprocess_box, slice_strip_image

    tensors, labels = [], []
    for entry in json.loads((SAMPLES_DIR / "manifest.json").read_text(encoding="utf-8")):
        with Image.open(SAMPLES_DIR / entry["filename"]) as img:
            boxes = slice_strip_image(img.copy(), num_boxes=len(entry["target_word"]))
        tensors += [preprocess_box(box)[0] for box in boxes]
        labels += list(entry["target_word"])
    return torch.stack(tensors), labels


def _reads(model: Any, batch: Any) -> list[str]:
    import torch

    from data.preprocess import CANONICAL_36_CHARS, fold_47_to_36_probs

    with torch.inference_mode():
        return [CANONICAL_36_CHARS[i] for i in fold_47_to_36_probs(torch.softmax(model(batch), dim=1)).argmax(dim=1).tolist()]


def _missing_weights() -> bool:
    return not all((WEIGHTS_DIR / name).is_file() for name in VISION_WEIGHTS)


def check_vision_models() -> CheckResult:
    """(b) Dense MLP and FP32 CNN load and read the committed sample boxes."""
    name = "(b) vision models classify the committed sample boxes"
    if _missing_weights():
        return CheckResult(name, "SKIP", DOWNLOAD_HINT)
    from models.char_models import load_char_cnn_fp32, load_dense_mlp

    batch, labels = _sample_boxes()
    acc = {key: sum(a == b for a, b in zip(_reads(model, batch), labels)) / len(labels)
           for key, model in (("cnn", load_char_cnn_fp32(WEIGHTS_DIR / "char_cnn_fp32.pt")), ("dense", load_dense_mlp(WEIGHTS_DIR / "dense_mlp.pt")))}
    ok = acc["cnn"] >= MIN_CNN_SAMPLE_ACC and acc["dense"] >= MIN_DENSE_SAMPLE_ACC
    return CheckResult(name, "PASS" if ok else "FAIL", f"{len(labels)} boxes: CNN {acc['cnn']:.1%} (min {MIN_CNN_SAMPLE_ACC:.0%}), Dense {acc['dense']:.1%} (min {MIN_DENSE_SAMPLE_ACC:.0%})")


def check_int8_cnn_portability() -> CheckResult:
    """(c) The INT8 checkpoint (saved on Windows/onednn) loads here and agrees with the FP32 CNN."""
    name = "(c) INT8 CNN loads on this platform and matches FP32"
    if _missing_weights():
        return CheckResult(name, "SKIP", DOWNLOAD_HINT)
    import torch

    from models.char_models import load_char_cnn_fp32
    from models.quantization import load_quantized_cnn

    batch, _ = _sample_boxes()
    fp32 = _reads(load_char_cnn_fp32(WEIGHTS_DIR / "char_cnn_fp32.pt"), batch)
    int8 = _reads(load_quantized_cnn(WEIGHTS_DIR / "char_cnn_int8.pt"), batch)
    agree = sum(a == b for a, b in zip(fp32, int8)) / len(fp32)
    detail = f"engine {torch.backends.quantized.engine}: {agree:.1%} of {len(fp32)} boxes agree (min {MIN_INT8_AGREEMENT:.0%})"
    return CheckResult(name, "PASS" if agree >= MIN_INT8_AGREEMENT else "FAIL", detail)


def _png_b64(width: int, height: int) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (width, height), 255).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def check_app_validation() -> CheckResult:
    """(d) Bad requests get a one-line 400 before any model is needed (the app's startup is not run)."""
    from fastapi.testclient import TestClient

    from app import app

    client = TestClient(app, base_url="http://127.0.0.1:8000")
    box = _png_b64(64, 64)
    cases = {
        "bad base64": {"sentence": "The [MASK] is hot.", "boxes_base64": ["!!!"]},
        "no blank": {"sentence": "No blank here.", "boxes_base64": [box]},
        "17 boxes": {"sentence": "The [MASK] is hot.", "boxes_base64": [box] * 17},
        "header bomb": {"sentence": "The [MASK] is hot.", "boxes_base64": [_png_b64(4097, 4)]},
        "wrong types": {"sentence": 7, "boxes_base64": "x"},
    }
    bad = [case for case, body in cases.items() if not _is_one_line_400(client.post("/api/recognize", json=body))]
    if not _is_one_line_400(client.get("/api/health", headers={"host": "evil.example"})):
        bad.append("foreign Host header")
    detail = f"{len(cases) + 1 - len(bad)}/{len(cases) + 1} bad requests rejected with a one-line 400" + (f"; wrong: {', '.join(bad)}" if bad else "")
    return CheckResult("(d) app input validation (no models loaded)", "FAIL" if bad else "PASS", detail)


def _is_one_line_400(resp: Any) -> bool:
    detail = resp.json().get("detail") if resp.status_code == 400 else None
    return isinstance(detail, str) and "\n" not in detail and "Traceback" not in detail


CHECKS: list[Callable[[], CheckResult]] = [check_recompute, check_vision_models, check_int8_cnn_portability, check_app_validation]


def _run(check: Callable[[], CheckResult]) -> CheckResult:
    try:
        return check()
    except Exception as exc:  # report, never crash: one failing claim must not hide the others
        return CheckResult(getattr(check, "__name__", "check"), "FAIL", f"{type(exc).__name__}: {' '.join(str(exc).split())}")


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    results = [_run(check) for check in CHECKS]
    for result in results:
        sys.stdout.write(f"{result.status:<4}  {result.name}: {result.detail}\n")
    failed = sum(r.status == "FAIL" for r in results)
    sys.stdout.write(f"{len(results) - failed}/{len(results)} checks without failure ({sum(r.status == 'SKIP' for r in results)} skipped)\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

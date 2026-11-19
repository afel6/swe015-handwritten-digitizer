"""HTTP tests for the optional full-page OCR routes (fake Ollama; the graded routes stay as they were)."""

from __future__ import annotations

import base64
import io
import threading
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from models import vlm_ocr
from models.assets import WEIGHT_FILES
from tests.fake_ollama import closed_port_url, fake_ollama

REPO_ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not all((REPO_ROOT / "weights" / n).is_file() for n in WEIGHT_FILES), reason="model weights not downloaded")


@pytest.fixture(scope="module")
def client() -> Any:
    from fastapi.testclient import TestClient

    from app import app

    with TestClient(app, base_url="http://127.0.0.1:8000") as test_client:
        yield test_client


def _png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _one_line_detail(resp: Any, code: int) -> str:
    assert resp.status_code == code, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, str) and detail and "\n" not in detail
    return detail


def test_ocr_status_without_ollama(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vlm_ocr, "OLLAMA_URL", closed_port_url())
    resp = client.get("/api/ocr/status")
    assert resp.status_code == 200 and resp.json()["available"] is False


def test_ocr_returns_transcription(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    with fake_ollama() as fake:
        monkeypatch.setattr(vlm_ocr, "OLLAMA_URL", fake.url)
        resp = client.post("/api/ocr", json={"image_base64": _png_b64()})
    assert resp.status_code == 200, resp.text
    assert resp.json()["text"] == "Invoice number: INV-2026-3706"


@pytest.mark.parametrize("body", [
    {"image_base64": "data:image/png;base64," + "A" * 8},
    {"image_base64": "not base64 at all!"},
    {"image_base64": base64.b64encode(b"GIF89a" + b"\x00" * 32).decode()},
    {"image_base64": "A" * 8_000_000},
    {"image_base64": "AAAA", "extra": 1},
    {"image_base64": 5},
])
def test_ocr_rejects_bad_images_without_calling_ollama(client: Any, monkeypatch: pytest.MonkeyPatch, body: dict) -> None:
    with fake_ollama() as fake:
        monkeypatch.setattr(vlm_ocr, "OLLAMA_URL", fake.url)
        resp = client.post("/api/ocr", json=body)
    _one_line_detail(resp, 400)
    assert fake.requests == []


def test_ocr_503_when_model_missing(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    with fake_ollama(models=[], status=404, reply={"error": "model not found"}) as fake:
        monkeypatch.setattr(vlm_ocr, "OLLAMA_URL", fake.url)
        resp = client.post("/api/ocr", json={"image_base64": _png_b64()})
    assert vlm_ocr.PULL_COMMAND in _one_line_detail(resp, 503)


def test_ocr_503_when_ollama_absent(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vlm_ocr, "OLLAMA_URL", closed_port_url())
    _one_line_detail(client.post("/api/ocr", json={"image_base64": _png_b64()}), 503)


def test_recognize_not_blocked_by_ocr(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    started, release = threading.Event(), threading.Event()

    def slow(_image: bytes) -> dict[str, Any]:
        started.set()
        release.wait(30)
        return {"text": "", "latency_ms": 0, "tokens": 0, "min_prob": None, "finish_reason": "stop"}

    monkeypatch.setattr(vlm_ocr, "transcribe", slow)
    worker = threading.Thread(target=lambda: client.post("/api/ocr", json={"image_base64": _png_b64()}))
    worker.start()
    try:
        assert started.wait(10)
        sample = client.get("/api/samples").json()[0]
        resp = client.post("/api/recognize", json={"sentence": sample["sentence"], "strip_base64": sample["strip_base64"], "num_boxes": sample["num_boxes"]})
        assert resp.status_code == 200 and not release.is_set()
    finally:
        release.set()
        worker.join(30)


def test_csp_allows_blob_previews(client: Any) -> None:
    assert "img-src 'self' data: blob:" in client.get("/").headers["content-security-policy"]


def test_second_ocr_while_busy_gets_quick_503(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    started, release = threading.Event(), threading.Event()

    def slow(_image: bytes) -> dict[str, Any]:
        started.set()
        release.wait(30)
        return {"text": "", "latency_ms": 0, "tokens": 0, "min_prob": None, "finish_reason": "stop"}

    monkeypatch.setattr(vlm_ocr, "transcribe", slow)
    worker = threading.Thread(target=lambda: client.post("/api/ocr", json={"image_base64": _png_b64()}))
    worker.start()
    try:
        assert started.wait(10)
        assert "Another OCR" in _one_line_detail(client.post("/api/ocr", json={"image_base64": _png_b64()}), 503)
    finally:
        release.set()
        worker.join(30)

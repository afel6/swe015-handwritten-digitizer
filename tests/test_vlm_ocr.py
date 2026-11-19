"""The local Ollama adapter behind the full-page OCR tab (tested against a fake server; one live test)."""

from __future__ import annotations

import io
import math

import pytest
from PIL import Image, ImageDraw, ImageFont

from models import vlm_ocr
from models.errors import UserInputError
from tests.fake_ollama import closed_port_url, fake_ollama


def _png(text: str = "", size: tuple[int, int] = (64, 32)) -> bytes:
    img = Image.new("RGB", size, "white")
    if text:
        ImageDraw.Draw(img).text((20, 20), text, fill="black", font=ImageFont.load_default(size=48))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_chat_sends_fixed_decoding_and_image() -> None:
    with fake_ollama() as fake:
        vlm_ocr.chat(_png(), vlm_ocr.OCR_PROMPT, base_url=fake.url)
    (req,) = fake.requests
    assert (req["model"], req["temperature"], req["seed"], req["logprobs"], req["stream"]) == (vlm_ocr.MODEL, 0, 42, True, False)
    image, text = req["messages"][0]["content"]
    assert image["image_url"]["url"].startswith("data:image/png;base64,iVBORw0KGgo")
    assert text["text"] == vlm_ocr.OCR_PROMPT


def test_transcribe_returns_text_and_confidence() -> None:
    with fake_ollama() as fake:
        got = vlm_ocr.transcribe(_png(), base_url=fake.url)
    assert got["text"] == "Invoice number: INV-2026-3706"
    assert got["tokens"] == 14
    assert got["min_prob"] == round(math.exp(-0.000002861027041944908), 4)
    assert got["finish_reason"] == "stop" and isinstance(got["latency_ms"], int)


def test_summarise_empty_reply() -> None:
    got = vlm_ocr.summarise({"choices": [{"message": {"content": ""}}]}, 3.0)
    assert (got["text"], got["tokens"], got["min_prob"]) == ("", 0, None)


def test_rejects_non_png_jpeg_without_calling_ollama() -> None:
    with fake_ollama() as fake, pytest.raises(UserInputError):
        vlm_ocr.transcribe(b"GIF89a....", base_url=fake.url)
    assert fake.requests == []


def test_status_reports_missing_model() -> None:
    with fake_ollama(models=[]) as fake:
        got = vlm_ocr.status(base_url=fake.url)
    assert (got["available"], got["model_ready"]) == (True, False)
    assert vlm_ocr.PULL_COMMAND in got["detail"]


def test_status_ready_with_model() -> None:
    with fake_ollama() as fake:
        assert vlm_ocr.status(base_url=fake.url)["model_ready"] is True


def test_status_when_ollama_absent() -> None:
    got = vlm_ocr.status(base_url=closed_port_url())
    assert (got["available"], got["model_ready"]) == (False, False)
    assert "ollama.com" in got["detail"]


def test_module_url_read_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    with fake_ollama() as fake:
        monkeypatch.setattr(vlm_ocr, "OLLAMA_URL", fake.url)
        assert vlm_ocr.status()["model_ready"] is True


@pytest.mark.parametrize("kwargs", [{"status": 500}, {"reply": b"not json"}, {"reply": {}}, {"reply": {"choices": []}}])
def test_transcribe_bad_reply_raises_unavailable(kwargs: dict) -> None:
    with fake_ollama(**kwargs) as fake, pytest.raises(vlm_ocr.OcrUnavailable) as info:
        vlm_ocr.transcribe(_png(), base_url=fake.url)
    assert "\n" not in str(info.value)


def test_transcribe_unreachable_raises_unavailable() -> None:
    with pytest.raises(vlm_ocr.OcrUnavailable):
        vlm_ocr.transcribe(_png(), base_url=closed_port_url())


@pytest.mark.skipif(not vlm_ocr.status()["model_ready"], reason="needs a local Ollama with qwen3-vl:8b-instruct-q4_K_M")
def test_live_ollama_reads_invoice_line() -> None:
    got = vlm_ocr.transcribe(_png("Invoice number: INV-2026-3706", (900, 90)))
    assert "INV-2026-3706" in got["text"]


def test_requests_bypass_any_configured_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(name, closed_port_url())
    import urllib.request

    monkeypatch.setattr(urllib.request, "_opener", None)  # urlopen caches its opener (and the proxies) on first use
    with fake_ollama() as fake:
        assert vlm_ocr.status(base_url=fake.url)["model_ready"] is True
        assert vlm_ocr.transcribe(_png(), base_url=fake.url)["text"] == "Invoice number: INV-2026-3706"
    assert len(fake.requests) == 1


@pytest.mark.parametrize("reply", [
    {"choices": [{"message": None}]},
    {"choices": [{"message": {"content": "x"}, "logprobs": {"content": ["not a dict"]}}]},
    {"choices": [{"message": {"content": "x"}, "logprobs": {"content": [{"logprob": 1000.0}]}}]},
])
def test_malformed_replies_raise_unavailable(reply: dict) -> None:
    with fake_ollama(reply=reply) as fake, pytest.raises(vlm_ocr.OcrUnavailable):
        vlm_ocr.transcribe(_png(), base_url=fake.url)


def test_non_http_listener_is_unavailable_not_a_crash() -> None:
    import socket
    import threading

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)

    def serve() -> None:
        for _ in range(2):
            conn, _addr = srv.accept()
            conn.recv(65536)
            conn.sendall(b"garbage that is not http\r\n\r\n")
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    url = f"http://127.0.0.1:{srv.getsockname()[1]}"
    try:
        assert vlm_ocr.status(base_url=url)["available"] is False
        with pytest.raises(vlm_ocr.OcrUnavailable):
            vlm_ocr.transcribe(_png(), base_url=url)
    finally:
        srv.close()

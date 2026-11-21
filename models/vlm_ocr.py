"""Full-page OCR through a local Ollama server running Qwen3-VL (optional; the graded pipeline never uses it).

Standard library only. The server address is fixed to this computer, so images never leave it.
"""

from __future__ import annotations

import base64
import http.client
import json
import math
import time
import urllib.error
import urllib.request
from typing import Any

from models.errors import UserInputError

__all__ = ["MODEL", "OCR_PROMPT", "OLLAMA_URL", "PULL_COMMAND", "OcrUnavailable", "chat", "status", "summarise", "transcribe"]

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen3-vl:8b-instruct-q4_K_M"
PULL_COMMAND = f"ollama pull {MODEL}"
INSTALL_HINT = f"Install Ollama from https://ollama.com/download, then run: {PULL_COMMAND}"
OCR_PROMPT = (
    "Transcribe the text in this image exactly as written. Do not correct, translate or explain. "
    "Keep digits in the script they are written in. Write ? for an unreadable character. "
    "Reply with an empty message if there is no text. "
    "Preserve mathematical notation, including Greek letters, operators and line breaks. "
    "Distinguish Greek theta from Latin O and zero by the written strokes."
)
TIMEOUT_S = 300.0
STATUS_TIMEOUT_S = 3.0
MAX_TOKENS = 1024
SEED = 42
_MS = 1000.0
_PROB_DP = 4
_IMAGE_MAGIC = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"))


# Never use a proxy: Ollama is on this computer, and images must not leave it (urllib would honour HTTP_PROXY).
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_TRANSPORT_ERRORS = (OSError, TimeoutError, http.client.HTTPException)


def open_url(req: urllib.request.Request | str, timeout_s: float) -> Any:
    """urlopen without any proxy."""
    return _OPENER.open(req, timeout=timeout_s)


class OcrUnavailable(RuntimeError):
    """Ollama or the model cannot answer; the message is one user-facing line."""


def _base(base_url: str | None) -> str:
    return (base_url or OLLAMA_URL).rstrip("/")


def _mime(image: bytes) -> str:
    for magic, mime in _IMAGE_MAGIC:
        if image.startswith(magic):
            return mime
    raise UserInputError("The image must be a PNG or JPEG.")


def status(base_url: str | None = None, timeout_s: float = STATUS_TIMEOUT_S) -> dict[str, Any]:
    """Whether Ollama answers and the OCR model is pulled. Never raises."""
    try:
        with open_url(f"{_base(base_url)}/api/tags", timeout_s) as resp:
            names = {m.get("name") for m in json.load(resp).get("models", [])}
    except (*_TRANSPORT_ERRORS, ValueError, AttributeError, TypeError):
        return {"available": False, "model_ready": False, "detail": f"Ollama is not running on this computer. {INSTALL_HINT}"}
    if MODEL not in names:
        return {"available": True, "model_ready": False, "detail": f"Ollama is running but the OCR model is missing. Run: {PULL_COMMAND}"}
    return {"available": True, "model_ready": True, "detail": f"Ready: {MODEL} on this computer."}


def _request_body(image: bytes, prompt: str, max_tokens: int) -> bytes:
    url = f"data:{_mime(image)};base64,{base64.b64encode(image).decode('ascii')}"
    body = {
        "model": MODEL, "temperature": 0, "seed": SEED, "max_tokens": max_tokens, "logprobs": True, "stream": False,
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": prompt}]}],
    }
    return json.dumps(body).encode("utf-8")


def chat(image: bytes, prompt: str, *, base_url: str | None = None, timeout_s: float = TIMEOUT_S, max_tokens: int = MAX_TOKENS) -> dict[str, Any]:
    """One /v1/chat/completions call; any transport or format problem becomes OcrUnavailable."""
    req = urllib.request.Request(f"{_base(base_url)}/v1/chat/completions", data=_request_body(image, prompt, max_tokens),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with open_url(req, timeout_s) as resp:
            reply = json.load(resp)
    except urllib.error.HTTPError as exc:
        hint = f" Run: {PULL_COMMAND}" if exc.code == 404 else ""
        raise OcrUnavailable(f"Ollama could not run the OCR model (HTTP {exc.code}).{hint}") from exc
    except _TRANSPORT_ERRORS as exc:
        raise OcrUnavailable(f"Ollama did not answer. {INSTALL_HINT}") from exc
    except ValueError as exc:
        raise OcrUnavailable("Ollama sent a reply that is not JSON.") from exc
    if not isinstance(reply, dict):
        raise OcrUnavailable("Ollama sent a reply without a transcription.")
    return reply


def summarise(reply: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
    """Text, latency, token count and the probability of the least certain token."""
    choice = reply["choices"][0]
    entries = (choice.get("logprobs") or {}).get("content") or []
    logprobs = [e["logprob"] for e in entries if isinstance(e.get("logprob"), (int, float))]
    return {
        "text": choice["message"].get("content") or "",
        "latency_ms": round(elapsed_ms),
        "tokens": len(logprobs),
        "min_prob": round(math.exp(min(logprobs)), _PROB_DP) if logprobs else None,
        "finish_reason": choice.get("finish_reason"),
    }


def transcribe(image: bytes, prompt: str = OCR_PROMPT, *, base_url: str | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    reply = chat(image, prompt, base_url=base_url)
    try:
        return summarise(reply, (time.perf_counter() - start) * _MS)
    except (KeyError, IndexError, TypeError, AttributeError, OverflowError, ValueError) as exc:
        raise OcrUnavailable("Ollama sent a reply without a usable transcription.") from exc

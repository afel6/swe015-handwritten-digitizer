"""A tiny stand-in for a local Ollama server, so OCR code can be tested without the real model."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from models.vlm_ocr import MODEL

# Raw /v1/chat/completions body for test line a, recorded from the real model on 2026-11-19.
REPLY_A: dict[str, Any] = json.loads(
    '{"id":"chatcmpl-281","object":"chat.completion","created":1795114573,"model":"qwen3-vl:8b-instruct-q4_K_M",'
    '"system_fingerprint":"fp_ollama","choices":[{"index":0,"message":{"role":"assistant","content":"Invoice number: INV-2026-3706"},'
    '"finish_reason":"stop","logprobs":{"content":[{"token":"Invoice","logprob":-1.1920930376163597e-7,"bytes":[73,110,118,111,105,99,101]},'
    '{"token":" number","logprob":0,"bytes":[32,110,117,109,98,101,114]},{"token":":","logprob":0,"bytes":[58]},'
    '{"token":" INV","logprob":-0.000002861027041944908,"bytes":[32,73,78,86]},{"token":"-","logprob":0,"bytes":[45]},'
    '{"token":"2","logprob":0,"bytes":[50]},{"token":"0","logprob":0,"bytes":[48]},{"token":"2","logprob":0,"bytes":[50]},'
    '{"token":"6","logprob":0,"bytes":[54]},{"token":"-","logprob":0,"bytes":[45]},{"token":"3","logprob":0,"bytes":[51]},'
    '{"token":"7","logprob":0,"bytes":[55]},{"token":"0","logprob":0,"bytes":[48]},{"token":"6","logprob":0,"bytes":[54]}]}}],'
    '"usage":{"prompt_tokens":1179,"prompt_tokens_details":{"cached_tokens":1178},"completion_tokens":15,"total_tokens":1194}}'
)


@dataclass
class FakeOllama:
    url: str
    requests: list[dict[str, Any]] = field(default_factory=list)


@contextmanager
def fake_ollama(reply: dict[str, Any] | bytes | None = None, status: int = 200, models: list[str] | None = None) -> Iterator[FakeOllama]:
    """Serve GET /api/tags and POST /v1/chat/completions on a free local port."""
    body = json.dumps(REPLY_A if reply is None else reply).encode() if not isinstance(reply, bytes) else reply
    tags = json.dumps({"models": [{"name": m, "digest": "70e58e37beb1"} for m in ([MODEL] if models is None else models)]}).encode()
    holder: dict[str, FakeOllama] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            return

        def _send(self, code: int, payload: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            self._send(200, tags) if self.path == "/api/tags" else self._send(404, b"{}")

        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            holder["fake"].requests.append(json.loads(raw))
            self._send(status, body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    holder["fake"] = FakeOllama(url=f"http://127.0.0.1:{server.server_address[1]}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield holder["fake"]
    finally:
        server.shutdown()
        server.server_close()


def closed_port_url() -> str:
    """A local URL nothing listens on."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = server.server_address[1]
    server.server_close()
    return f"http://127.0.0.1:{port}"

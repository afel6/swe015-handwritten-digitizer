"""Handwritten Answer Digitizer web app: `python app.py` serves the UI and API on http://127.0.0.1:8000 (CPU only)."""

from __future__ import annotations

import base64
import binascii
import math
import os
import re
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WEIGHTS_DIR = Path(os.environ.get("SWE015_WEIGHTS_DIR", ROOT / "weights"))
RESULTS_DIR = ROOT / "results"

from models.assets import missing_assets, missing_assets_message  # noqa: E402

if __name__ == "__main__" and (_missing := missing_assets(WEIGHTS_DIR, RESULTS_DIR)):  # fail fast, before torch/transformers load
    sys.stderr.write(missing_assets_message(_missing) + "\n")
    raise SystemExit(1)

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

from data.prepare_data import read_data  # noqa: E402
from data.preprocess import (  # noqa: E402
    MAX_IMAGE_BYTES,
    MAX_IMAGE_DIM,
    load_validated_image_bytes,
    slice_strip_image,
)
from models.cloze import normalize_masked_sentence  # noqa: E402
from models.recognizer import RecognitionService  # noqa: E402

HOST, PORT = "127.0.0.1", 8000
STATIC_DIR, SAMPLES_DIR = ROOT / "static", ROOT / "data" / "sample_worksheets"
MAX_BODY_BYTES = 8 * 1024 * 1024
ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost"})  # blocks DNS rebinding: other sites' names never reach the API
MIN_BOXES, MAX_BOXES = 1, 16
MAX_B64_CHARS = math.ceil(MAX_IMAGE_BYTES / 3) * 4
_SAMPLE_NAME_RE = re.compile(r"^sample_\d{2}\.png$")
_SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
}
_BAD_REQUEST = "Send JSON with a text 'sentence' and either 'boxes_base64' (list of base64 images) or 'strip_base64' (base64 image) plus optional integer 'num_boxes'."
_IMAGE_RULES = f"Each image must be a PNG or JPEG of at most {MAX_IMAGE_BYTES // (1024 * 1024)} MB and {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM} pixels."


class RecognizeRequest(BaseModel):
    """Request body; strict types and no unknown fields."""

    model_config = ConfigDict(extra="forbid", strict=True)
    sentence: str
    boxes_base64: list[str] | None = None
    strip_base64: str | None = None
    num_boxes: int | None = None


def _load_samples() -> list[dict[str, Any]]:
    """Samples come only from files the manifest names; nothing from the request is ever used as a path."""
    out = []
    for entry in read_data(SAMPLES_DIR / "manifest.json", as_json=True):
        name = str(entry["filename"])
        if not _SAMPLE_NAME_RE.fullmatch(name):
            continue
        raw = (SAMPLES_DIR / name).read_bytes()
        out.append({"id": entry["id"], "sentence": entry["sentence"], "target_word": entry["target_word"], "difficulty": entry["difficulty"],
                    "num_boxes": len(str(entry["target_word"])), "strip_base64": base64.b64encode(raw).decode("ascii")})
    return out


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    missing = missing_assets(WEIGHTS_DIR, RESULTS_DIR)
    if missing:
        raise RuntimeError(missing_assets_message(missing))
    app.state.service = RecognitionService(WEIGHTS_DIR, RESULTS_DIR)
    app.state.samples = _load_samples()
    yield


app = FastAPI(title="Handwritten Answer Digitizer", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def guard_and_headers(request: Request, call_next: Any) -> Any:
    host = request.headers.get("host", "").rsplit(":", 1)[0].strip("[]").lower()
    if host not in ALLOWED_HOSTS:
        return _secure(JSONResponse({"detail": f"Unknown host; open the app at http://{HOST}:{PORT}."}, status_code=400))
    if request.method == "POST":
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            return _secure(JSONResponse({"detail": "Send the request body as application/json."}, status_code=400))
        length = request.headers.get("content-length")
        if length is None or not length.isdigit() or int(length) > MAX_BODY_BYTES:
            return _secure(JSONResponse({"detail": f"Request body must declare its size and be at most {MAX_BODY_BYTES // (1024 * 1024)} MB."}, status_code=400))
    return _secure(await call_next(request))


def _secure(response: Any) -> Any:
    for key, value in _SECURITY_HEADERS.items():
        response.headers[key] = value
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, _exc: RequestValidationError) -> JSONResponse:
    return JSONResponse({"detail": _BAD_REQUEST}, status_code=400)


@app.exception_handler(Exception)
async def unexpected_error(_request: Request, _exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": "Internal error while processing the request."}, status_code=500)


def _bad(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _decode_image(b64: str) -> Image.Image:
    if len(b64) > MAX_B64_CHARS:
        raise _bad(_IMAGE_RULES)
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _bad("Images must be plain base64 strings (no 'data:' prefix).") from exc
    try:
        return load_validated_image_bytes(raw)
    except (ValueError, OSError, Image.DecompressionBombError) as exc:
        raise _bad(_IMAGE_RULES) from exc


def _box_images(body: RecognizeRequest) -> list[Image.Image]:
    if (body.boxes_base64 is None) == (body.strip_base64 is None):
        raise _bad("Send exactly one of 'boxes_base64' or 'strip_base64'.")
    if body.boxes_base64 is not None:
        if not MIN_BOXES <= len(body.boxes_base64) <= MAX_BOXES:
            raise _bad(f"Send between {MIN_BOXES} and {MAX_BOXES} boxes.")
        return [_decode_image(b) for b in body.boxes_base64]
    if body.num_boxes is not None and not MIN_BOXES <= body.num_boxes <= MAX_BOXES:
        raise _bad(f"num_boxes must be between {MIN_BOXES} and {MAX_BOXES}.")
    strip = _decode_image(str(body.strip_base64))
    try:
        boxes = slice_strip_image(strip, num_boxes=body.num_boxes)
    except ValueError as exc:
        raise _bad("Could not find the character boxes in the strip image.") from exc
    if not MIN_BOXES <= len(boxes) <= MAX_BOXES:
        raise _bad(f"The strip must contain between {MIN_BOXES} and {MAX_BOXES} boxes.")
    return boxes


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/static/app.js", include_in_schema=False)
def script() -> FileResponse:
    return FileResponse(STATIC_DIR / "app.js", media_type="text/javascript")


@app.get("/static/app.css", include_in_schema=False)
def stylesheet() -> FileResponse:
    return FileResponse(STATIC_DIR / "app.css", media_type="text/css")


@app.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    return {"status": "ok", "models_loaded": getattr(request.app.state, "service", None) is not None, "device": "cpu"}


@app.get("/api/samples")
def samples(request: Request) -> list[dict[str, Any]]:
    return request.app.state.samples


@app.post("/api/recognize")
def recognize(body: RecognizeRequest, request: Request) -> dict[str, Any]:
    """Plain def: runs in the threadpool so CPU-bound inference never blocks the event loop."""
    try:
        sentence = normalize_masked_sentence(body.sentence)
    except ValueError as exc:
        message = str(exc)
        raise _bad(message[:1].upper() + message[1:]) from exc
    images = _box_images(body)
    try:
        return request.app.state.service.recognize(sentence, images)
    except ValueError as exc:
        raise _bad(str(exc)) from exc


def main() -> int:
    missing = missing_assets(WEIGHTS_DIR, RESULTS_DIR)
    if missing:
        sys.stderr.write(missing_assets_message(missing) + "\n")
        return 1
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""HTTP tests for the FastAPI app: health, samples, recognition, and one-line 400s for every bad input."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from models.assets import WEIGHT_FILES

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = REPO_ROOT / "data" / "sample_worksheets"
REQUIRED = [REPO_ROOT / "weights" / n for n in WEIGHT_FILES]
pytestmark = pytest.mark.skipif(not all(p.is_file() for p in REQUIRED), reason="model weights not downloaded")


@pytest.fixture(scope="module")
def client() -> Any:
    from fastapi.testclient import TestClient

    from app import app

    with TestClient(app, base_url="http://127.0.0.1:8000") as test_client:
        yield test_client


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _sample(index: int = 0) -> dict[str, Any]:
    entry = json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))[index]
    return {**entry, "strip_base64": base64.b64encode((SAMPLES / entry["filename"]).read_bytes()).decode("ascii")}


def _boxes_b64(index: int = 0) -> tuple[dict[str, Any], list[str]]:
    from data.preprocess import slice_strip_image

    sample = _sample(index)
    strip = Image.open(io.BytesIO(base64.b64decode(sample["strip_base64"])))
    return sample, [_png_b64(box) for box in slice_strip_image(strip, num_boxes=len(sample["target_word"]))]


def _assert_one_line_400(resp: Any) -> str:
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, str) and detail and "\n" not in detail
    for leak in ("Traceback", "File \"", ".py", "site-packages", "C:\\", "/home/", "pydantic", "errors.pydantic.dev"):
        assert leak not in detail, detail
    return detail


# ---------------------------------------------------------------- health, samples, page


def test_health_reports_models_loaded_on_cpu(client: Any) -> None:
    assert client.get("/api/health").json() == {"status": "ok", "models_loaded": True, "device": "cpu"}


def test_samples_list_all_manifest_strips(client: Any) -> None:
    samples = client.get("/api/samples").json()
    assert len(samples) == 15
    for s in samples:
        assert {"id", "sentence", "target_word", "difficulty", "num_boxes", "strip_base64"} <= set(s)
        assert s["num_boxes"] == len(s["target_word"])
        Image.open(io.BytesIO(base64.b64decode(s["strip_base64"], validate=True))).verify()


def test_index_page_and_assets_are_served_with_security_headers(client: Any) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "default-src 'self'" in page.headers["content-security-policy"] and page.headers["x-content-type-options"] == "nosniff"
    assert "access-control-allow-origin" not in page.headers
    for asset, kind in (("/static/app.js", "javascript"), ("/static/app.css", "css")):
        resp = client.get(asset)
        assert resp.status_code == 200 and kind in resp.headers["content-type"]
    script = client.get("/static/app.js").text
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in script  # server strings reach the page only through textContent
    assert "<script>" not in page.text and "https://" not in page.text  # no inline script, no CDN
    assert client.get("/static/../app.py").status_code == 404


# ---------------------------------------------------------------- recognition


def _check_response(body: dict[str, Any], n_boxes: int) -> None:
    assert len(body["boxes"]) == n_boxes and body["request_id"]
    for box in body["boxes"]:
        assert {"index", "empty", "preview_png_b64", "top2", "dense_pred", "cnn_fp32_pred", "cnn_int8_pred", "gradcam_png_b64"} <= set(box)
        assert Image.open(io.BytesIO(base64.b64decode(box["preview_png_b64"]))).size == (28, 28)
        assert len(box["top2"]) == 2 and box["top2"][0]["prob"] >= box["top2"][1]["prob"]
    names = [m["name"] for m in body["models"]]
    assert names == ["raw_fp32_cnn", "classical_ngram", "bert_fp32", "quantized_app"]
    for row in body["models"]:
        assert {"name", "precision", "predicted_word", "raw_word", "top_candidates", "latency_ms"} <= set(row)
        assert len(row["top_candidates"]) <= 5 and row["latency_ms"] > 0
    assert body["primary_model"] == "quantized_app"
    assert "INT8" in body["models"][3]["precision"]


def test_recognize_strip_runs_all_four_models(client: Any) -> None:
    sample = _sample(1)
    resp = client.post("/api/recognize", json={"sentence": sample["sentence"], "strip_base64": sample["strip_base64"], "num_boxes": len(sample["target_word"])})
    assert resp.status_code == 200, resp.text
    _check_response(resp.json(), len(sample["target_word"]))


def test_digits_only_reads_strip_without_sentence_context(client: Any) -> None:
    sample = _sample()
    response = client.post("/api/recognize", json={
        "sentence": "", "strip_base64": sample["strip_base64"],
        "num_boxes": len(sample["target_word"]), "digits_only": True,
    })
    assert response.status_code == 200
    result = response.json()
    assert result["primary_model"] == "digits_only"
    assert result["models"][0]["predicted_word"] == sample["target_word"]


def test_recognize_boxes_matches_strip_result(client: Any) -> None:
    sample, boxes = _boxes_b64(1)
    via_boxes = client.post("/api/recognize", json={"sentence": sample["sentence"], "boxes_base64": boxes}).json()
    via_strip = client.post("/api/recognize", json={"sentence": sample["sentence"], "strip_base64": sample["strip_base64"], "num_boxes": len(boxes)}).json()
    _check_response(via_boxes, len(boxes))
    assert [m["predicted_word"] for m in via_boxes["models"]] == [m["predicted_word"] for m in via_strip["models"]]


def test_blank_boxes_are_marked_empty_and_left_out_of_the_word(client: Any) -> None:
    sample, boxes = _boxes_b64(1)
    blank = _png_b64(Image.new("L", (64, 64), 255))
    body = client.post("/api/recognize", json={"sentence": sample["sentence"], "boxes_base64": [*boxes, blank]}).json()
    assert body["boxes"][-1]["empty"] is True and body["boxes"][-1]["gradcam_png_b64"] is None
    assert len(body["models"][0]["raw_word"]) == len(boxes)
    _assert_one_line_400(client.post("/api/recognize", json={"sentence": sample["sentence"], "boxes_base64": [blank, blank]}))


# ---------------------------------------------------------------- bad input -> one-line 400


def _big_png_header_bomb() -> str:
    return _png_b64(Image.new("1", (4097, 8), 1))


def _bad_bodies() -> list[tuple[str, Any]]:
    sample, boxes = _boxes_b64(0)
    good = sample["sentence"]
    huge = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * (5 * 1024 * 1024 + 10)).decode("ascii")
    return [
        ("bad base64", {"sentence": good, "boxes_base64": ["!!!not base64!!!"]}),
        ("data url prefix", {"sentence": good, "boxes_base64": ["data:image/png;base64," + boxes[0]]}),
        ("not an image", {"sentence": good, "boxes_base64": [base64.b64encode(b"hello world").decode()]}),
        ("over 5 MB", {"sentence": good, "boxes_base64": [huge]}),
        ("header dims bomb", {"sentence": good, "boxes_base64": [_big_png_header_bomb()]}),
        ("zero boxes", {"sentence": good, "boxes_base64": []}),
        ("seventeen boxes", {"sentence": good, "boxes_base64": [boxes[0]] * 17}),
        ("both inputs", {"sentence": good, "boxes_base64": boxes, "strip_base64": sample["strip_base64"]}),
        ("neither input", {"sentence": good}),
        ("num_boxes zero", {"sentence": good, "strip_base64": sample["strip_base64"], "num_boxes": 0}),
        ("num_boxes seventeen", {"sentence": good, "strip_base64": sample["strip_base64"], "num_boxes": 17}),
        ("no mask", {"sentence": "No blank here.", "boxes_base64": boxes}),
        ("two masks", {"sentence": "A [MASK] and ___ here.", "boxes_base64": boxes}),
        ("501 chars", {"sentence": "A [MASK] " + "x" * 492, "boxes_base64": boxes}),
        ("sentence not text", {"sentence": 123, "boxes_base64": boxes}),
        ("boxes not a list", {"sentence": good, "boxes_base64": boxes[0]}),
        ("num_boxes as text", {"sentence": good, "strip_base64": sample["strip_base64"], "num_boxes": "3"}),
        ("unknown field", {"sentence": good, "boxes_base64": boxes, "path": "../../etc/passwd"}),
    ]


@pytest.mark.parametrize("case", range(18))
def test_bad_inputs_return_one_line_400(client: Any, case: int) -> None:
    name, body = _bad_bodies()[case]
    detail = _assert_one_line_400(client.post("/api/recognize", json=body))
    assert len(detail) < 200, name


def test_sentence_errors_keep_the_mask_token_spelling(client: Any) -> None:
    _, boxes = _boxes_b64(0)
    detail = _assert_one_line_400(client.post("/api/recognize", json={"sentence": "No blank here.", "boxes_base64": boxes}))
    assert "[MASK]" in detail and detail[0].isupper()


def test_malformed_json_and_oversized_body_return_400(client: Any) -> None:
    _assert_one_line_400(client.post("/api/recognize", content=b"{not json", headers={"content-type": "application/json"}))
    _assert_one_line_400(client.post("/api/recognize", content=b"x" * (8 * 1024 * 1024 + 1), headers={"content-type": "application/json"}))


@pytest.mark.parametrize("host", ["evil.example", "127.0.0.1.evil.example", "attacker.localhost.example"])
def test_foreign_host_headers_are_rejected(client: Any, host: str) -> None:
    resp = client.get("/api/health", headers={"host": host})
    assert resp.status_code == 400 and resp.json() == {"detail": "Unknown host; open the app at http://127.0.0.1:8000."}


@pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000", "localhost", "127.0.0.1"])
def test_local_host_headers_are_accepted(client: Any, host: str) -> None:
    assert client.get("/api/health", headers={"host": host}).status_code == 200


@pytest.mark.parametrize("ctype", ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"])
def test_recognize_accepts_only_json_bodies(client: Any, ctype: str) -> None:
    sample, boxes = _boxes_b64(0)
    body = json.dumps({"sentence": sample["sentence"], "boxes_base64": boxes}).encode()
    detail = _assert_one_line_400(client.post("/api/recognize", content=body, headers={"content-type": ctype}))
    assert "application/json" in detail


def test_int8_bert_loaded_from_file_matches_fresh_quantization(client: Any) -> None:
    service = client.app.state.service
    fresh = service.bert_fp32.to_int8()
    sentence, words = "The [MASK] is the closest star to the Earth.", ["sun", "moon", "planet"]
    loaded = service.bert_int8.span_log_probs(sentence, words)
    rebuilt = fresh.span_log_probs(sentence, words)
    assert all(loaded[w] == pytest.approx(rebuilt[w], abs=1e-5) for w in words)




# ---------------------------------------------------------------- request limits and error isolation


def test_transfer_encoding_cannot_bypass_the_body_cap(client: Any) -> None:
    resp = client.post("/api/recognize", content=b'{"sentence": "x"}', headers={"content-type": "application/json", "content-length": "16", "transfer-encoding": "chunked"})
    assert resp.status_code == 400 and resp.json()["detail"].startswith("Request body must declare its size")


def test_a_body_of_exactly_the_cap_is_not_refused_for_size(client: Any) -> None:
    prefix, suffix = b'{"sentence": "', b'", "boxes_base64": ["x"]}'
    body = prefix + b"a" * (8 * 1024 * 1024 - len(prefix) - len(suffix)) + suffix
    assert len(body) == 8 * 1024 * 1024
    detail = _assert_one_line_400(client.post("/api/recognize", content=body, headers={"content-type": "application/json"}))
    assert not detail.startswith("Request body must declare its size")


def test_sixteen_boxes_and_num_boxes_sixteen_pass_the_count_checks(client: Any) -> None:
    sample, boxes = _boxes_b64(1)
    assert client.post("/api/recognize", json={"sentence": sample["sentence"], "boxes_base64": (boxes * 16)[:16]}).status_code == 200
    resp = client.post("/api/recognize", json={"sentence": sample["sentence"], "strip_base64": sample["strip_base64"], "num_boxes": 16})
    assert "between 1 and 16" not in resp.json().get("detail", "")


def test_num_boxes_out_of_range_is_refused_even_with_boxes(client: Any) -> None:
    sample, boxes = _boxes_b64(1)
    detail = _assert_one_line_400(client.post("/api/recognize", json={"sentence": sample["sentence"], "boxes_base64": boxes, "num_boxes": 99}))
    assert "num_boxes" in detail


def test_library_errors_never_reach_the_client(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise ValueError("boom\nC:\\secret\\path.py")

    from fastapi.testclient import TestClient

    monkeypatch.setattr(client.app.state.service, "recognize", boom)
    sample, boxes = _boxes_b64(1)
    production_like = TestClient(client.app, base_url="http://127.0.0.1:8000", raise_server_exceptions=False)
    resp = production_like.post("/api/recognize", json={"sentence": sample["sentence"], "boxes_base64": boxes})
    assert resp.status_code == 500 and resp.json() == {"detail": "Internal error while processing the request."}


def test_recognition_runs_one_request_at_a_time(client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    import time

    state = {"active": 0, "peak": 0}
    guard = threading.Lock()

    def slow(*_a: Any, **_k: Any) -> dict[str, Any]:
        with guard:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.2)
        with guard:
            state["active"] -= 1
        return {"request_id": "x", "boxes": [], "models": [], "primary_model": "quantized_app"}

    monkeypatch.setattr(client.app.state.service, "recognize", slow)
    sample, boxes = _boxes_b64(1)
    threads = [threading.Thread(target=lambda: client.post("/api/recognize", json={"sentence": sample["sentence"], "boxes_base64": boxes})) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert state["peak"] == 1

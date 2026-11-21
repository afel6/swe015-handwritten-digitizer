"""Static checks on the two-tab UI (the browser behaviour is checked by hand in the browser pane)."""

from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "static"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_index_has_both_tabs() -> None:
    html = _read("index.html")
    for needle in ('id="tab-reader"', 'id="tab-ocr"', 'id="panel-reader"', 'id="panel-ocr"', 'role="tablist"', '<script src="/static/ocr.js?v='):
        assert needle in html, needle


def test_ocr_script_targets_api_and_downscales() -> None:
    js = _read("ocr.js")
    assert "/api/ocr" in js and "/api/ocr/status" in js
    assert "MAX_SIDE_PX = 2048" in js and "JPEG_QUALITY = 0.9" in js
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in js  # server text reaches the page only through textContent


def test_strip_mode_hint_explains_equal_boxes() -> None:
    assert "one word written in equal boxes" in _read("index.html")


def test_strip_upload_can_open_the_original_file_in_ocr() -> None:
    html = _read("index.html")
    assert 'id="strip-open-ocr" type="button"' in html
    assert "Use this image in Full-page OCR" in html
    assert 'aria-describedby="strip-advice-text"' in html
    assert "openInFullPageOcr(state.strip.file)" in _read("app.js")
    assert 'setTool("ocr", true);\n  return pick(file);' in _read("ocr.js")


def test_ocr_panel_has_setup_card_and_copy() -> None:
    html = _read("index.html")
    assert 'id="ocr-setup"' in html and 'id="ocr-copy"' in html and 'dir="auto"' in html
    assert "ollama pull qwen3-vl:8b-instruct-q4_K_M" in html


def test_motion_respects_reduced_motion() -> None:
    assert "prefers-reduced-motion" in _read("app.css")


def test_ocr_script_is_served() -> None:
    import app

    paths = {getattr(r, "path", "") for r in app.app.routes}
    assert "/static/ocr.js" in paths


def test_stale_ocr_responses_are_ignored() -> None:
    js = _read("ocr.js")
    assert "ocr.requestId" in js and "if (id !== ocr.requestId) return;" in js

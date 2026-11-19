"""Tests for the grader-facing tools: scripts/download_assets.py and verify.py."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

import verify
from scripts import download_assets

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = REPO_ROOT / "weights"


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _asset(tmp_path: Path, files: dict[str, bytes]) -> download_assets.Asset:
    payload = _zip_bytes(files)
    return download_assets.Asset("test", "https://example.invalid/a.zip", hashlib.sha256(payload).hexdigest(),
                                 {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}), payload


def test_download_extracts_and_verifies_every_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asset, payload = _asset(tmp_path, {"weights/a.pt": b"alpha", "weights/b.pkl": b"beta"})
    monkeypatch.setattr(download_assets, "_fetch", lambda url: payload)
    download_assets.install(asset, tmp_path)
    assert (tmp_path / "weights" / "a.pt").read_bytes() == b"alpha"
    monkeypatch.setattr(download_assets, "_fetch", lambda url: pytest.fail("files already present must not be downloaded again"))
    download_assets.install(asset, tmp_path)


def test_checksum_mismatch_is_rejected_and_nothing_is_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asset, _ = _asset(tmp_path, {"weights/a.pt": b"alpha"})
    monkeypatch.setattr(download_assets, "_fetch", lambda url: _zip_bytes({"weights/a.pt": b"tampered"}))
    with pytest.raises(ValueError, match="sha256"):
        download_assets.install(asset, tmp_path)
    assert not (tmp_path / "weights").exists()


@pytest.mark.parametrize("name", ["../evil.txt", "/abs/evil.txt", "weights/../../evil.txt", "C:/evil.txt"])
def test_unsafe_zip_paths_are_rejected(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _zip_bytes({name: b"x"})
    asset = download_assets.Asset("bad", "https://example.invalid/b.zip", hashlib.sha256(payload).hexdigest(), {name: hashlib.sha256(b"x").hexdigest()})
    monkeypatch.setattr(download_assets, "_fetch", lambda url: payload)
    with pytest.raises(ValueError, match="unsafe"):
        download_assets.install(asset, tmp_path / "root")


def test_cli_reports_failures_in_one_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def offline(url: str) -> bytes:
        raise OSError("network unreachable")

    monkeypatch.setattr(download_assets, "_fetch", offline)
    monkeypatch.setattr(download_assets, "ASSETS", [download_assets.Asset("v9", "https://example.invalid/v9.zip", "0" * 64, {"weights/x.pt": "0" * 64})])
    assert download_assets.main(["--root", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "v9" in err


def test_release_assets_pin_every_required_file() -> None:
    names = {n for a in download_assets.ASSETS for n in a.files}
    assert {"weights/dense_mlp.pt", "weights/char_cnn_fp32.pt", "weights/char_cnn_int8.pt", "weights/ngram_lm.pkl", "weights/bert_int8_dynamic.pt"} <= names
    assert all(a.url.startswith("https://github.com/afel6/swe015-handwritten-digitizer/releases/download/") for a in download_assets.ASSETS)


def test_recompute_check_passes_on_committed_results() -> None:
    assert verify.check_recompute().status == "PASS"


def test_weight_checks_skip_with_the_download_command_when_weights_are_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify, "WEIGHTS_DIR", tmp_path)
    for check in (verify.check_vision_models(), verify.check_int8_cnn_portability()):
        assert check.status == "SKIP" and "python scripts/download_assets.py" in check.detail


@pytest.mark.skipif(not (WEIGHTS / "char_cnn_int8.pt").is_file(), reason="vision weights not downloaded")
def test_weight_checks_pass_with_real_weights() -> None:
    assert verify.check_vision_models().status == "PASS"
    assert verify.check_int8_cnn_portability().status == "PASS"


def test_app_validation_check_runs_without_models() -> None:
    result = verify.check_app_validation()
    assert result.status == "PASS", result.detail


def test_main_exit_code_reflects_failures(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    ok, skip, bad = verify.CheckResult("a", "PASS", ""), verify.CheckResult("b", "SKIP", "why"), verify.CheckResult("c", "FAIL", "boom")
    monkeypatch.setattr(verify, "CHECKS", [lambda: ok, lambda: skip])
    assert verify.main([]) == 0
    monkeypatch.setattr(verify, "CHECKS", [lambda: ok, lambda: bad])
    assert verify.main([]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "boom" in out


def test_readme_grader_section_matches_agents_md() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("<!-- grader section: keep identical to AGENTS.md -->\n", 1)[1].split("<!-- end grader section -->", 1)[0]
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert section.strip() in agents and "python verify.py" in section
    for command in ("python scripts/recompute_metrics.py", "python scripts/download_assets.py", "python app.py", "python -m pytest -q"):
        assert command in section


def test_readme_tables_match_the_reported_results() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    report = (REPO_ROOT / "results" / "benchmark_results.md").read_text(encoding="utf-8")
    rows = [line for line in report.splitlines() if line.startswith("| (") or line.startswith("| char_") or line.startswith("| dense_")]
    assert rows and all(row in readme for row in rows)


def test_readme_architecture_is_generated_from_the_shared_spec() -> None:
    from scripts import architecture

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "```mermaid\n" + architecture.mermaid() + "\n```" in readme
    keys = {n.key for n in architecture.NODES}
    assert all(e.src in keys and e.dst in keys for e in architecture.EDGES)
    assert [n.key for n in architecture.NODES if n.primary] == ["B8"]


def test_readme_ai_usage_matches_the_slide_text() -> None:
    from scripts import slide_content

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert all(line in readme for line in slide_content.ai_usage_lines("Abdalraouf Alkabir"))


def test_pyproject_declares_the_tested_python_range() -> None:
    import tomllib

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["requires-python"] == ">=3.12,<3.14"


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_a: Any) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.data if size < 0 else self.data[:size]


def test_downloads_larger_than_the_cap_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(download_assets, "MAX_DOWNLOAD_BYTES", 10)
    monkeypatch.setattr(download_assets.urllib.request, "urlopen", lambda *_a, **_k: _FakeResponse(b"x" * 11))
    with pytest.raises(ValueError, match="larger than"):
        download_assets._fetch("https://example.invalid/big.zip")


def test_interrupted_download_is_one_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import http.client

    def broken(*_a: Any, **_k: Any) -> Any:
        raise http.client.IncompleteRead(b"partial")

    monkeypatch.setattr(download_assets.urllib.request, "urlopen", broken)
    assert download_assets.main(["--root", str(tmp_path)]) == 1
    assert capsys.readouterr().err.count("\n") == 1


def test_slides_cli_reports_missing_results_in_one_line(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import generate_slides

    monkeypatch.setattr(generate_slides.sc, "RESULTS", tmp_path / "missing")
    monkeypatch.setattr(generate_slides.render, "__defaults__", (generate_slides.OUT_DIR, tmp_path / "missing"))
    assert generate_slides.main(["--out-dir", str(tmp_path / "out")]) == 1
    assert capsys.readouterr().err.count("\n") == 1


def test_slides_cli_names_a_corrupt_cnn_checkpoint(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import generate_slides

    bad = tmp_path / "char_cnn_fp32.pt"
    bad.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(generate_slides, "GRADCAM_WEIGHTS", bad)
    assert generate_slides.main(["--out-dir", str(tmp_path / "out")]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "char_cnn_fp32.pt" in err and "python scripts/download_assets.py" in err and "results are incomplete" not in err


def test_architecture_edges_follow_the_recognizer() -> None:
    from scripts import architecture

    edges = {(e.src, e.dst): e.label for e in architecture.EDGES}
    assert ("CNN32", "OUT") in edges and ("CNN8", "OUT") in edges and ("MLP", "OUT") in edges
    assert "empty" not in next(n.label for n in architecture.NODES if n.key == "PRE")
    assert "non-empty" in edges[("FOLD32", "POOL32")] and "non-empty" in edges[("FOLD8", "POOL8")]


def test_readme_states_venv_runtime_and_timing_protocol() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("<!-- grader section: keep identical to AGENTS.md -->\n", 1)[1].split("<!-- end grader section -->", 1)[0]
    assert "python -m venv .venv" in section and r".venv\Scripts\Activate.ps1" in section and "source .venv/bin/activate" in section
    assert "a few minutes once the weights are downloaded" in section
    assert "100 timed runs after 10 warm-up runs" in readme and "median of 3" in readme
    assert "one item is 0.33 pp" in readme


def test_benchmark_report_states_the_timing_protocol() -> None:
    report = (REPO_ROOT / "results" / "benchmark_results.md").read_text(encoding="utf-8")
    assert "100 timed runs after 10 warm-up runs" in report and "mean over the 300 test worksheets" in report


def test_failure_examples_carry_the_reported_latencies() -> None:
    failures = json.loads((REPO_ROOT / "results" / "failure_examples.json").read_text(encoding="utf-8"))
    rows = json.loads((REPO_ROOT / "results" / "test_predictions.json").read_text(encoding="utf-8"))["rows"]
    by_item = {key: {r["item_id"]: r["latency_ms"] for r in recs} for key, recs in rows.items()}
    for failure in failures:
        for key, latency in failure["latency_ms"].items():
            assert latency == by_item[key][failure["id"]], (failure["id"], key)


def test_hardware_line_says_a_cuda_build_was_not_used_for_timing() -> None:
    from scripts import slide_content

    res = {"benchmark_results": {"hardware": {"cpu_model": "X", "torch_threads": 4, "ram_gb": 8.0, "torch": "2.9.1+cu128", "transformers": "5"}}}
    assert "CUDA build, GPU unused for timing" in slide_content.hardware_line(res)


def _reference() -> dict:
    return json.loads((REPO_ROOT / "results" / "vlm_reference.json").read_text(encoding="utf-8"))


def test_readme_explains_two_parts_and_reference() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Two parts" in readme and "Full-page OCR" in readme and "not trained by us" in readme
    for run in _reference()["runs"].values():
        assert f"{run['word_acc_pct']:.2f}" in readme
    assert "results/vlm_reference.json" in readme.split("<!-- grader section: keep identical to AGENTS.md -->\n", 1)[1]


def test_reference_line_reads_json() -> None:
    from scripts import slide_content

    res = slide_content.load_results()
    line = slide_content.reference_line(res)
    assert "not trained by us" in line and "Qwen3-VL" in line
    for run in _reference()["runs"].values():
        assert f"{run['word_acc_pct']:.2f}" in line

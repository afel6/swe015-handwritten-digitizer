"""The Qwen3-VL reference benchmark: prompts, scoring, report shape and recompute (no Ollama needed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import benchmark_vlm as bv
from scripts import recompute_metrics as rm

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
ITEMS = [{"id": f"t{i}", "sentence": "The soup is [MASK].", "target_word": "hot"} for i in range(3)]
REPLIES = ["hot", " Hot.", "HAT"]


def _records(mode: str = "strip_only") -> list[dict[str, Any]]:
    replies = iter(REPLIES)

    def ask(_image: bytes, _prompt: str) -> dict[str, Any]:
        return {"text": next(replies), "latency_ms": 10, "tokens": 1, "min_prob": 1.0, "finish_reason": "stop"}

    return bv.run_mode(ITEMS, [b"png"] * 3, mode, ask)


def _report() -> dict[str, Any]:
    meta = {"digest": "70e58e37beb1", "ollama_version": "0.40.16", "gpu": "Test GPU"}
    return bv.build_report({mode: _records(mode) for mode in bv.MODES}, meta)


def test_prompts_differ_by_mode() -> None:
    sentence = "The soup is [MASK]."
    assert sentence in bv.build_prompt("strip_sentence", sentence)
    assert sentence not in bv.build_prompt("strip_only", sentence)
    with pytest.raises(ValueError):
        bv.build_prompt("other", sentence)


def test_run_mode_normalises_and_scores() -> None:
    records = _records()
    assert [r["predicted_word"] for r in records] == ["HOT", "HOT", "HAT"]
    assert records[1]["raw_reply"] == " Hot." and records[0]["item_id"] == "t0"
    run = _report()["runs"]["strip_only"]
    assert run["word_acc_pct"] == pytest.approx(200 / 3)
    assert run["char_acc_pct"] == pytest.approx(800 / 9)
    assert run["latency_mean_ms"] == pytest.approx(10.0)


def test_report_is_labelled_as_outside_reference() -> None:
    report = _report()
    assert report["label"] == bv.LABEL == "Reference, not trained by us, GPU"
    assert report["model"] == "qwen3-vl:8b-instruct-q4_K_M" and report["n"] == 3
    assert set(report["runs"]) == set(bv.MODES) and report["settings"]["temperature"] == 0


def test_compare_reference_matches_report() -> None:
    report = _report()
    checks = rm.compare_reference(report)
    assert checks and all(ok for *_, ok in checks)
    report["runs"]["strip_only"]["word_acc_pct"] = 99.0
    assert not all(ok for *_, ok in rm.compare_reference(report))


def test_recompute_skips_missing_reference(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--predictions", str(RESULTS / "test_predictions.json"), "--results", str(RESULTS / "benchmark_results.json")]
    assert rm.main([*args, "--reference", str(tmp_path / "missing.json")]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("35/35 ")


def test_recompute_includes_reference_checks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(_report()), encoding="utf-8")
    args = ["--predictions", str(RESULTS / "test_predictions.json"), "--results", str(RESULTS / "benchmark_results.json")]
    assert rm.main([*args, "--reference", str(ref)]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("43/43 ")


@pytest.mark.skipif(not (RESULTS / "vlm_reference.json").is_file(), reason="reference run not committed yet")
def test_committed_reference_recomputes() -> None:
    report = json.loads((RESULTS / "vlm_reference.json").read_text(encoding="utf-8"))
    assert report["label"] == bv.LABEL and report["n"] == 300
    assert all(len(report["runs"][m]["records"]) == 300 for m in bv.MODES)
    assert all(ok for *_, ok in rm.compare_reference(report))

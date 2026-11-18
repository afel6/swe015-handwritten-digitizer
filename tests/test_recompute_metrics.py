"""Tests for the torch-free metric recomputation (scripts/recompute_metrics.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import recompute_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent


def _row(pairs: list[tuple[str, str]], in_pool: list[bool] | None, latency: float = 10.0) -> list[dict[str, Any]]:
    return [{"item_id": f"t{i}", "target_word": t, "raw_word": p, "predicted_word": p, "char_correct": 0, "char_total": len(t), "latency_ms": latency,
             "in_pool": None if in_pool is None else in_pool[i]} for i, (t, p) in enumerate(pairs)]


def _fixture(tmp_path: Path, tamper: float = 0.0) -> tuple[Path, Path]:
    pairs = [("SUN", "sun"), ("MOON", "M00N"), ("2000", "20O0"), ("STAR", "STAR")]
    rows = {"raw_fp32": _row(pairs, None, 5.0), "classical": _row(pairs, [True, True, False, True], 15.0), "quantized_app": _row(pairs, [True, False, False, True], 20.0)}
    vision = {"n": 10, "class_totals_47": [10], "cnn": {"correct_per_class_47": [3, 4, 1], "correct_36": 9}}
    results = {"ladder_n": 4, "vision_n": 10, "recall_pct": {"fp32": 75.0, "int8": 50.0}, "vision": {"cnn": {"acc_47_pct": 80.0 + tamper, "acc_36_pct": 90.0}},
               "ladder": {k: {"word_acc_pct": 50.0, "char_acc_pct": 100.0 * 12 / 15, "latency_mean_ms": lat} for k, lat in (("raw_fp32", 5.0), ("classical", 15.0), ("quantized_app", 20.0))}}
    pred_path, res_path = tmp_path / "p.json", tmp_path / "r.json"
    pred_path.write_text(json.dumps({"rows": rows, "vision": vision}), encoding="utf-8")
    res_path.write_text(json.dumps(results), encoding="utf-8")
    return pred_path, res_path


def test_recomputed_numbers_match_and_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pred, res = _fixture(tmp_path)
    assert recompute_metrics.main(["--predictions", str(pred), "--results", str(res)]) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out and "15/15 numbers match" in out


def test_a_misreported_number_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pred, res = _fixture(tmp_path, tamper=0.02)
    assert recompute_metrics.main(["--predictions", str(pred), "--results", str(res)]) == 1
    assert "FAIL  vision cnn acc_47_pct" in capsys.readouterr().out


def test_char_accuracy_is_recomputed_not_copied() -> None:
    rec = _row([("MOON", "M00N")], [True])
    rec[0]["char_correct"] = 4  # a wrong stored value must not be trusted
    assert recompute_metrics.row_metrics(rec)["char_acc_pct"] == pytest.approx(50.0)


def test_missing_file_is_one_line_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert recompute_metrics.main(["--predictions", str(tmp_path / "nope.json")]) == 1
    assert capsys.readouterr().err.count("\n") == 1


@pytest.mark.skipif(not (REPO_ROOT / "results" / "test_predictions.json").is_file(), reason="no committed test predictions yet")
def test_committed_results_recompute_exactly() -> None:
    assert recompute_metrics.main(["--predictions", str(REPO_ROOT / "results" / "test_predictions.json"), "--results", str(REPO_ROOT / "results" / "benchmark_results.json")]) == 0

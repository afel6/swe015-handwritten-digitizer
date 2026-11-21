from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.evaluate_worksheet_pilot import evaluate


def _write(path: Path, rows: list[tuple]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    return path


def test_pilot_compares_paired_times_and_keeps_leading_zero(tmp_path: Path) -> None:
    reference = _write(tmp_path / "reference.csv", [
        ("page_id", "question", "kind", "answer"),
        ("p1", 1, "digits", "004"),
        ("p1", 2, "word", "Moss"),
        ("p1", 3, "digits", ""),
        ("p1", 4, "word", "ICE"),
    ])
    header = ("page_id", "question", "method", "kind", "suggestion", "transcription", "seconds")
    assisted = _write(tmp_path / "assisted.csv", [
        (*header, "suggestion_status"),
        ("p1", 1, "assisted", "digits", "4", "004", 10, "ok"),
        ("p1", 2, "assisted", "word", "MOSS", "moss", 10, "ok"),
        ("p1", 3, "assisted", "digits", "", "", 10, "ok"),
        ("p1", 4, "assisted", "word", "ICF", "ICE", 10, "ok"),
    ])
    manual = _write(tmp_path / "manual.csv", [
        header,
        ("p1", 1, "manual", "digits", "", "004", 17),
        ("p1", 2, "manual", "word", "", "MOSS", 17),
        ("p1", 3, "manual", "digits", "", "", 17),
        ("p1", 4, "manual", "word", "", "ICF", 17),
    ])
    result = evaluate(reference, [assisted, manual])
    assert result["median_paired_seconds_saved"] == 7
    assert result["digits"]["assisted_suggestion_errors"] == 1
    assert result["words"]["manual_final_errors"] == 1
    assert result["all"]["assisted_final_errors"] == 0
    with pytest.raises(ValueError, match="Missing manual results"):
        evaluate(reference, [assisted])


@pytest.mark.parametrize("has_status", [True, False], ids=["recorded-failure", "legacy-empty"])
def test_pilot_does_not_credit_unverified_blank_suggestions(tmp_path: Path, has_status: bool) -> None:
    answers = [(1, "digits", ""), (2, "word", ""), (3, "digits", "004"), (4, "word", "ICE")]
    reference = _write(tmp_path / "reference.csv", [
        ("page_id", "question", "kind", "answer"),
        *(("p1", question, kind, answer) for question, kind, answer in answers),
    ])
    rows = [
        ("page_id", "question", "method", "kind", "suggestion", "transcription", "seconds", "suggestion_status"),
        ("p1", 1, "assisted", "digits", "", "", 10, "failed"),
        ("p1", 2, "assisted", "word", "", "", 10, "ok"),
        ("p1", 3, "assisted", "digits", "004", "004", 10, "ok"),
        ("p1", 4, "assisted", "word", "ICE", "ICE", 10, "ok"),
        *(("p1", question, "manual", kind, "", answer, 17, "") for question, kind, answer in answers),
    ]
    exports = _write(tmp_path / "exports.csv", rows if has_status else [row[:-1] for row in rows])

    result = evaluate(reference, [exports])
    assert result["all"]["assisted_suggestion_errors"] == (1 if has_status else 2)
    assert result["all"]["assisted_suggestion_accuracy_pct"] == (75 if has_status else 50)
    assert result["all"]["assisted_recognition_failures"] == (1 if has_status else 0)
    assert result["all"]["assisted_unverified_empty_suggestions"] == (0 if has_status else 2)
    assert result["all"]["assisted_final_errors"] == 0
    assert result["all"]["manual_final_errors"] == 0
    assert result["median_paired_seconds_saved"] == 7

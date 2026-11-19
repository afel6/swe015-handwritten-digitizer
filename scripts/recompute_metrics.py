"""Recompute every reported benchmark number from results/test_predictions.json (pure Python: no torch, no models).

For each ladder row it re-derives word accuracy, character accuracy (position-wise matches over the
target length), mean latency and candidate recall from the per-item records; for each vision model it
re-derives the 47-class and 36-folded accuracies from per-class counts. Every number is compared with
results/benchmark_results.json to within 0.01 and printed as PASS/FAIL. Exit code 0 only if all match.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PREDICTIONS = Path("results/test_predictions.json")
RESULTS = Path("results/benchmark_results.json")
REFERENCE = Path("results/vlm_reference.json")
TOLERANCE = 0.01
_TOKEN_RE = re.compile(r"[A-Z0-9]+")
_RECALL_ROW = {"fp32": "classical", "int8": "quantized_app"}


def normalize(word: str) -> str:
    """Same rule as models.classical_nlp.normalize_word: uppercase, alphanumerics only."""
    return "".join(_TOKEN_RE.findall(str(word).upper()))


def row_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Word acc %, char acc %, mean latency (ms) and, when known, candidate recall % for one ladder row."""
    if not records:
        raise ValueError("A ladder row has no records.")
    hits = chars = total_chars = 0
    for rec in records:
        pred, target = normalize(rec["predicted_word"]), normalize(rec["target_word"])
        hits += pred == target
        chars += sum(a == b for a, b in zip(pred, target))
        total_chars += len(target)
    out = {"word_acc_pct": 100.0 * hits / len(records), "char_acc_pct": 100.0 * chars / total_chars,
           "latency_mean_ms": sum(float(r["latency_ms"]) for r in records) / len(records)}
    if all(r.get("in_pool") is not None for r in records):
        out["recall_pct"] = 100.0 * sum(bool(r["in_pool"]) for r in records) / len(records)
    return out


def vision_metrics(counts: Mapping[str, Any], name: str) -> dict[str, float]:
    n = int(counts["n"])
    return {"acc_47_pct": 100.0 * sum(counts[name]["correct_per_class_47"]) / n, "acc_36_pct": 100.0 * int(counts[name]["correct_36"]) / n}


def compare(predictions: Mapping[str, Any], results: Mapping[str, Any]) -> list[tuple[str, float, float, bool]]:
    """(label, recomputed, reported, ok) for every reported number that the predictions determine."""
    checks: list[tuple[str, float, float, bool]] = []

    def add(label: str, got: float, want: float) -> None:
        checks.append((label, got, float(want), abs(got - float(want)) <= TOLERANCE))

    rows = predictions["rows"]
    for key, reported in results["ladder"].items():
        if key not in rows:
            checks.append((f"ladder {key} present in predictions", 0.0, 1.0, False))
            continue
        got = row_metrics(rows[key])
        for metric in ("word_acc_pct", "char_acc_pct", "latency_mean_ms"):
            add(f"ladder {key} {metric}", got[metric], reported[metric])
        add(f"ladder {key} n", float(len(rows[key])), reported["n"])
    for pool, key in _RECALL_ROW.items():
        if key in rows:
            add(f"candidate recall {pool}", row_metrics(rows[key])["recall_pct"], results["recall_pct"][pool])
    if "raw_fp32" in rows:
        add("strip-path char acc (raw FP32 CNN)", row_metrics(rows["raw_fp32"])["char_acc_pct"], results["strip_char_acc_pct"])
    for name in results["vision"]:
        got = vision_metrics(predictions["vision"], name)
        for metric in ("acc_47_pct", "acc_36_pct"):
            add(f"vision {name} {metric}", got[metric], results["vision"][name][metric])
    add("ladder N", float(len(next(iter(rows.values())))), results["ladder_n"])
    add("vision N", float(predictions["vision"]["n"]), results["vision_n"])
    return checks


def compare_reference(reference: Mapping[str, Any]) -> list[tuple[str, float, float, bool]]:
    """Checks for the Qwen3-VL reference (an outside model, reported apart from the ladder)."""
    checks: list[tuple[str, float, float, bool]] = []
    for mode, run in reference["runs"].items():
        got = row_metrics(run["records"])
        for metric in ("word_acc_pct", "char_acc_pct", "latency_mean_ms"):
            checks.append((f"reference {mode} {metric}", got[metric], float(run[metric]), abs(got[metric] - float(run[metric])) <= TOLERANCE))
        checks.append((f"reference {mode} n", float(len(run["records"])), float(reference["n"]), len(run["records"]) == reference["n"]))
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute the reported benchmark numbers from the committed per-item predictions.")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--reference", type=Path, default=REFERENCE, help="Qwen3-VL reference results; checked only when the file exists")
    args = parser.parse_args(argv)
    try:
        predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
        results = json.loads(args.results.read_text(encoding="utf-8"))
        checks = compare(predictions, results)
        if args.reference.is_file():
            checks += compare_reference(json.loads(args.reference.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        sys.stderr.write(f"Error: could not recompute metrics: {' '.join(str(exc).split())}\n")
        return 1
    for label, got, want, ok in checks:
        sys.stdout.write(f"{'PASS' if ok else 'FAIL'}  {label}: recomputed {got:.4f} vs reported {want:.4f}\n")
    failed = sum(not ok for *_, ok in checks)
    sys.stdout.write(f"{len(checks) - failed}/{len(checks)} numbers match within {TOLERANCE}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

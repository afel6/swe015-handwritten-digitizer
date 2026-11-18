"""Timing-only re-measure of the reported benchmark (python scripts/evaluate_all.py --retime).

Runs the same models, params, dev warm-up and test items as the reported run, several times. Every
pass must reproduce the committed per-item outputs exactly (raw word, prediction, candidate-pool hit,
character counts) and the same vision accuracies; on any difference it stops before writing anything.
Only latency fields change. The pass with the median app-model latency is reported (all passes are kept),
so per-item latencies and the reported means stay consistent. The previous timings are preserved once in
results/benchmark_run3_timing_superseded.json.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from data.prepare_data import (
    load_emnist_dataset,
    load_worksheet_items,
    read_data,
    write_data,
)
from data.word_pipeline import item_to_strip
from scripts import bench_report
from scripts import evaluate_all as ev

SUPERSEDED = Path("results/benchmark_run3_timing_superseded.json")
OUTPUT_KEYS = ("item_id", "target_word", "raw_word", "predicted_word", "char_correct", "char_total", "in_pool")
REASON = ("Run 3 timings were measured in a throttled power state (INT8 CNN batch-1 slower than FP32, app model 177 ms/word "
          "vs 110 ms in run 2). The benchmark was re-timed on AC power in Windows 'Best performance' mode with nothing else "
          "running. Accuracy is unchanged: every per-item output of every re-timing pass was verified identical to run 3.")


def _outputs(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {key: [{k: rec[k] for k in OUTPUT_KEYS} for rec in recs] for key, recs in rows.items()}


def check_identical(committed: Mapping[str, Any], fresh_rows: Mapping[str, Any], fresh_vision: Mapping[str, Any], reported_vision: Mapping[str, Any]) -> None:
    """Raise ValueError naming the first row/item whose output differs from the committed run."""
    old, new = _outputs(committed["rows"]), _outputs(fresh_rows)
    for key, recs in old.items():
        if key not in new or len(new[key]) != len(recs):
            raise ValueError(f"Re-timed row {key} does not match the committed predictions; nothing was written.")
        bad = next((a["item_id"] for a, b in zip(recs, new[key]) if a != b), None)
        if bad is not None:
            raise ValueError(f"Re-timed output differs from the committed predictions (row {key}, item {bad}); nothing was written.")
    for name, row in reported_vision.items():
        if abs(fresh_vision[name]["acc_47_pct"] - row["acc_47_pct"]) > 1e-9 or abs(fresh_vision[name]["acc_36_pct"] - row["acc_36_pct"]) > 1e-9:
            raise ValueError(f"Re-timed vision accuracy for {name} differs from the reported run; nothing was written.")


def _timing_pass(items: Sequence[Mapping[str, Any]], strips: Sequence[Any], models: Mapping[str, Any], emnist_root: Path) -> dict[str, Any]:
    ladder, records = ev.word_ladder(items, strips, models)
    vision, _ = ev.vision_table(models, emnist_root)
    _, bert_lat = ev.bert_standalone(models)
    return {"ladder": ladder, "rows": ev.prediction_records(records), "vision": vision, "bert_latency_ms": bert_lat}


def _summary(p: Mapping[str, Any]) -> dict[str, Any]:
    return {"ladder_latency_mean_ms": {k: v["latency_mean_ms"] for k, v in p["ladder"].items()},
            "vision_latency_ms": {k: {"b1": v["latency_b1_ms"], "b8": v["latency_b8_ms"]} for k, v in p["vision"].items()}, "bert_latency_ms": p["bert_latency_ms"]}


def _superseded(results: Mapping[str, Any], reason: str = REASON) -> dict[str, Any]:
    return {"reason": reason, "power_state": results.get("hardware", {}).get("power", "not recorded in run 3"),
            "ladder_latency_mean_ms": {k: v["latency_mean_ms"] for k, v in results["ladder"].items()},
            "vision_latency_ms": {k: {"b1": v["latency_b1_ms"], "b8": v["latency_b8_ms"]} for k, v in results["vision"].items()},
            "bert_latency_ms": results["bert_latency_ms"]}


def _merge(results: Mapping[str, Any], committed: Mapping[str, Any], chosen: Mapping[str, Any], passes: list[dict[str, Any]], index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    new = dict(results)
    new["ladder"] = {k: {**v, "latency_mean_ms": chosen["ladder"][k]["latency_mean_ms"]} for k, v in results["ladder"].items()}
    new["vision"] = {k: {**v, "latency_b1_ms": chosen["vision"][k]["latency_b1_ms"], "latency_b8_ms": chosen["vision"][k]["latency_b8_ms"]} for k, v in results["vision"].items()}
    new["bert_latency_ms"] = chosen["bert_latency_ms"]
    new["hardware"] = bench_report.hardware_metadata()
    new["retime"] = {"passes": passes, "reported_pass": index, "rule": "pass with the median app-model (row 4) mean latency"}
    preds = dict(committed)
    preds["rows"] = {k: [{**old, "latency_ms": fresh["latency_ms"]} for old, fresh in zip(recs, chosen["rows"][k], strict=True)] for k, recs in committed["rows"].items()}
    return new, preds


def retime(worksheets: Path, passes: int, emnist_root: Path = ev.EMNIST_ROOT, superseded: Path | None = None, reason: str = REASON) -> dict[str, Any]:
    """Re-measure latencies `passes` times; write only if every pass reproduces the committed outputs.

    The timings being replaced are kept once in `superseded` (default SUPERSEDED) with `reason`.
    """
    superseded = SUPERSEDED if superseded is None else Path(superseded)
    if passes < 1:
        raise ValueError("--passes must be at least 1.")
    torch.set_num_threads(ev.THREADS)
    committed, results = read_data(ev.OUT["predictions"], as_json=True), read_data(ev.OUT["json"], as_json=True)
    items = load_worksheet_items(worksheets, required=ev.REQUIRED_ITEM_KEYS)
    ev.check_splits(items, worksheets, ev.expected_split(worksheets))
    warmup = load_worksheet_items(ev.DEV_WORKSHEETS_PATH, required=ev.REQUIRED_ITEM_KEYS)[: ev.WARMUP_ITEMS]
    ev.check_splits(warmup, ev.DEV_WORKSHEETS_PATH, "val")
    models = ev.load_models(ev.load_fusion_params(ev.PARAMS_PATH), save_int8=False)
    train_ds, test_ds = load_emnist_dataset(emnist_root, train=True), load_emnist_dataset(emnist_root, train=False)
    for item in warmup:
        ev.run_item(item, item_to_strip(item, train_ds), models)
    strips = [item_to_strip(it, test_ds if it["emnist_split"] == "test" else train_ds) for it in items]
    runs = []
    for _ in range(passes):
        run = _timing_pass(items, strips, models, emnist_root)
        check_identical(committed, run["rows"], run["vision"], results["vision"])
        runs.append(run)
    order = sorted(range(passes), key=lambda i: runs[i]["ladder"]["quantized_app"]["latency_mean_ms"])
    index = order[(passes - 1) // 2]
    new_results, new_preds = _merge(results, committed, runs[index], [_summary(r) for r in runs], index)
    if not superseded.is_file():
        write_data(superseded, _superseded(results, reason), as_json=True)
    write_data(ev.OUT["json"], new_results, as_json=True)
    write_data(ev.OUT["md"], bench_report.render_markdown(new_results))
    write_data(ev.OUT["predictions"], new_preds, as_json=True)
    if ev.OUT["failures"].is_file():
        failures = read_data(ev.OUT["failures"], as_json=True)
        write_data(ev.OUT["failures"], bench_report.refresh_failure_latencies(failures, new_preds["rows"]), as_json=True)
    return new_results

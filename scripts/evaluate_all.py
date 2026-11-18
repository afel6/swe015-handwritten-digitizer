"""End-to-end CPU benchmark: vision models (Table 1) and the word-correction ladder (Table 2).

Run it alone on the machine. All timing is on CPU with torch.set_num_threads(4) and time.perf_counter().
The test worksheets are read once: one pass gives both accuracy and per-item latency. The per-item
predictions are written to results/test_predictions.json so scripts/recompute_metrics.py can
re-derive every reported number without torch or the models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

if str(_ROOT := Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from data._io_errors import describe_cli_os_error
from data.emnist_data import extract_oriented_tensors
from data.prepare_data import (
    DEV_WORKSHEETS_PATH,
    EMNIST_ROOT,
    TEST_WORKSHEETS_PATH,
    load_emnist_dataset,
    load_worksheet_items,
    read_data,
    write_data,
)
from data.preprocess import fold_47_to_36_probs
from data.word_pipeline import item_to_strip, strip_to_probs
from models.bert_reranker import BertMLMReranker
from models.char_models import NUM_CLASSES, load_char_cnn_fp32, load_dense_mlp
from models.classical_nlp import ClassicalNLPCorrector, normalize_word
from models.ladder import run_ladder, timed
from models.quantization import (
    benchmark_cpu_latency,
    get_model_disk_size_mb,
    load_quantized_cnn,
    save_quantized_state_dict,
)
from scripts import bench_report
from scripts.train_vision import BATCH_SIZE, FOLD_TARGET_MAP, make_latency_batches

THREADS = 4
WARMUP_ITEMS = 10
LATENCY_WARMUP = bench_report.LATENCY_WARMUP
LATENCY_RUNS = bench_report.LATENCY_RUNS
LATENCY_BATCH_SIZES = (1, 8)
EXPECTED_TUNING_FILE = DEV_WORKSHEETS_PATH.as_posix()
REQUIRED_ITEM_KEYS = ("id", "sentence", "target_word", "emnist_split", "emnist_indices")
PARAM_KEYS = ("alpha", "lambda_vis", "lambda_bert", "gamma")
WEIGHTS = {"mlp": Path("weights/dense_mlp.pt"), "cnn_fp32": Path("weights/char_cnn_fp32.pt"), "cnn_int8": Path("weights/char_cnn_int8.pt"), "ngram": Path("weights/ngram_lm.pkl"), "bert_int8": Path("weights/bert_int8_dynamic.pt")}
OUT = {"json": Path("results/benchmark_results.json"), "md": Path("results/benchmark_results.md"), "failures": Path("results/failure_examples.json"), "predictions": Path("results/test_predictions.json")}
PARAMS_PATH = Path("results/bert_fusion_params.json")
_LABELS = {"raw_fp32": "(1) Trained NN, no correction: FP32 CNN", "raw_int8": "(1b) parity: INT8 CNN, no correction", "classical": "(2) Classical: FP32 CNN + SymSpell + n-gram fusion",
           "bert_fp32": "(3) Transformer: FP32 CNN + same pool + FP32 BERT", "quantized_app": "(4) Quantized Transformer [App]: INT8 CNN pool + INT8 BERT", "ablation_fp32cnn_int8bert": "(ablation) FP32 CNN + same pool + INT8 BERT"}
_POOL_OF_ROW = {"raw_fp32": None, "raw_int8": None, "classical": "fp32", "bert_fp32": "fp32", "quantized_app": "int8", "ablation_fp32cnn_int8bert": "fp32"}
_VISION_MODELS = (("dense_mlp_fp32", "mlp"), ("char_cnn_fp32", "cnn_fp32"), ("char_cnn_int8", "cnn_int8"))


def load_fusion_params(path: Path = PARAMS_PATH) -> dict[str, float]:
    """Read the tuned BERT fusion params and check they were tuned on the dev worksheets."""
    doc = read_data(Path(path), as_json=True)
    params = doc.get("params") if isinstance(doc, Mapping) else None
    if not isinstance(params, Mapping) or any(not isinstance(params.get(k), (int, float)) or isinstance(params.get(k), bool) for k in PARAM_KEYS):
        raise ValueError(f"{Path(path).as_posix()} must contain 'params' with numeric {', '.join(PARAM_KEYS)}; rerun python scripts/tune_bert_fusion.py.")
    if doc.get("tuned_on") != EXPECTED_TUNING_FILE:
        raise ValueError(f"Fusion params in {Path(path).as_posix()} were tuned on {doc.get('tuned_on')!r}, expected {EXPECTED_TUNING_FILE!r}.")
    return {k: float(params[k]) for k in PARAM_KEYS}


def expected_split(path: Path) -> str | None:
    """EMNIST split a worksheet file must use: 'test' for the held-out file, 'val' for dev, else either."""
    if Path(path).resolve() == Path(TEST_WORKSHEETS_PATH).resolve():
        return "test"
    if Path(path).resolve() == Path(DEV_WORKSHEETS_PATH).resolve():
        return "val"
    return None


def check_splits(items: Sequence[Mapping[str, Any]], path: Path, expected: str | None) -> None:
    """Fail before any model loads if an item would draw glyphs from the wrong EMNIST split."""
    allowed = {expected} if expected else {"test", "val"}
    bad = next((it for it in items if it.get("emnist_split") not in allowed), None)
    if bad is not None:
        wanted = " or ".join(repr(s) for s in sorted(allowed))
        raise ValueError(f"Item {bad.get('id')} in {Path(path).as_posix()} has emnist_split {bad.get('emnist_split')!r}; every item must use {wanted}.")


def load_models(params: Mapping[str, float], save_int8: bool) -> dict[str, Any]:
    """Load every artifact once; the INT8 BERT file is written only when results are saved."""
    bert_fp32 = BertMLMReranker()
    bert_int8 = bert_fp32.to_int8()
    if save_int8:
        save_quantized_state_dict(bert_int8.model, WEIGHTS["bert_int8"], metadata={"base_model": bert_fp32.model_name, "quantization": "dynamic_int8_linear"})
    return {"mlp": load_dense_mlp(WEIGHTS["mlp"]), "cnn_fp32": load_char_cnn_fp32(WEIGHTS["cnn_fp32"]), "cnn_int8": load_quantized_cnn(WEIGHTS["cnn_int8"]),
            "ngram": ClassicalNLPCorrector.load(WEIGHTS["ngram"]), "bert_fp32": bert_fp32, "bert_int8": bert_int8, "params": dict(params)}


def _vision_predictions(model: torch.nn.Module, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        probs = torch.cat([torch.softmax(model(chunk), dim=1) for chunk in images.split(BATCH_SIZE)])
    return probs.argmax(1).numpy(), fold_47_to_36_probs(probs).argmax(1).numpy()


def vision_table(models: Mapping[str, Any], emnist_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """EMNIST balanced test accuracy (47-class and 36-folded), disk size and CPU latency, plus per-class counts."""
    images, labels = extract_oriented_tensors(load_emnist_dataset(emnist_root, train=False))
    y47, y36 = labels.numpy(), FOLD_TARGET_MAP[labels].numpy()
    table, counts = {}, {"n": int(labels.numel()), "class_totals_47": np.bincount(y47, minlength=NUM_CLASSES).tolist()}
    for name, key in _VISION_MODELS:
        p47, p36 = _vision_predictions(models[key], images)
        ok47, ok36 = p47 == y47, p36 == y36
        counts[name] = {"correct_per_class_47": np.bincount(y47[ok47], minlength=NUM_CLASSES).tolist(), "correct_36": int(ok36.sum())}
        lat = {b: benchmark_cpu_latency(models[key], make_latency_batches(images, b, LATENCY_WARMUP + LATENCY_RUNS), warmup=LATENCY_WARMUP, runs=LATENCY_RUNS)["mean_ms"] for b in LATENCY_BATCH_SIZES}
        table[name] = {"acc_47_pct": 100.0 * float(ok47.mean()), "acc_36_pct": 100.0 * float(ok36.mean()), "disk_mb": get_model_disk_size_mb(WEIGHTS[key]), "latency_b1_ms": lat[1], "latency_b8_ms": lat[8]}
    return table, counts


def run_item(item: Mapping[str, Any], strip: Any, models: Mapping[str, Any]) -> dict[str, Any]:
    """Run every ladder row on one strip (models.ladder.run_ladder); returns outputs plus per-row end-to-end latency (ms)."""
    sentence, n = str(item["sentence"]), len(str(item["target_word"]))
    p32, t32 = timed(lambda: strip_to_probs(strip, models["cnn_fp32"], num_boxes=n))
    p8, t8 = timed(lambda: strip_to_probs(strip, models["cnn_int8"], num_boxes=n))
    ladder = run_ladder(sentence, p32, p8, models, {"fp32": t32, "int8": t8})
    outputs = {k: row["predicted_word"] for k, row in ladder["rows"].items()}
    target = normalize_word(str(item["target_word"]))
    return {"id": item["id"], "sentence": sentence, "target": target, "outputs": outputs, "latency_ms": {k: row["latency_ms"] for k, row in ladder["rows"].items()},
            "raw_fp32": outputs["raw_fp32"], "raw_int8": outputs["raw_int8"], "target_in_pool_fp32": target in ladder["pools"]["fp32"],
            "target_in_pool_int8": target in ladder["pools"]["int8"], "numeric_preserved_int8": ladder["numeric_preserved_int8"],
            "mean_conf_int8": ladder["mean_conf_int8"]}


def row_sizes(models: Mapping[str, Any]) -> dict[str, float]:
    """Sum of every artifact each pipeline loads (MB); both BERTs are sized from the models loaded for this run."""
    cnn32, cnn8, lm = (get_model_disk_size_mb(WEIGHTS[k]) for k in ("cnn_fp32", "cnn_int8", "ngram"))
    bert32, bert8 = get_model_disk_size_mb(models["bert_fp32"].model), get_model_disk_size_mb(models["bert_int8"].model)
    return {"raw_fp32": cnn32, "raw_int8": cnn8, "classical": cnn32 + lm, "bert_fp32": cnn32 + lm + bert32, "quantized_app": cnn8 + lm + bert8, "ablation_fp32cnn_int8bert": cnn32 + lm + bert8, "bert_fp32_only": bert32, "bert_int8_only": bert8}


def word_ladder(items: Sequence[Mapping[str, Any]], strips: Sequence[Any], models: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [run_item(it, st, models) for it, st in zip(items, strips, strict=True)]
    sizes, n_chars = row_sizes(models), sum(len(r["target"]) for r in records)
    ladder = {}
    for key, label in _LABELS.items():
        scores = [bench_report.word_scores(r["outputs"][key], r["target"]) for r in records]
        ladder[key] = {"label": label, **bench_report.summarize_row([s[0] for s in scores], [s[1] for s in scores], n_chars, [r["latency_ms"][key] for r in records], sizes[key])}
    return ladder, records


def prediction_records(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Per row and item: everything recompute_metrics.py needs (no images)."""
    rows: dict[str, list[dict[str, Any]]] = {}
    for key, pool in _POOL_OF_ROW.items():
        raw_key = "raw_int8" if key in ("raw_int8", "quantized_app") else "raw_fp32"
        rows[key] = [{"item_id": r["id"], "target_word": r["target"], "raw_word": r[raw_key], "predicted_word": r["outputs"][key],
                      "char_correct": bench_report.word_scores(r["outputs"][key], r["target"])[1], "char_total": len(r["target"]),
                      "latency_ms": r["latency_ms"][key], "in_pool": None if pool is None else r[f"target_in_pool_{pool}"]} for r in records]
    return rows


def bert_standalone(models: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    sizes = row_sizes(models)
    sentence = "The [MASK] is the closest star to the Earth."
    lat = {k: benchmark_cpu_latency(lambda s, r=models[f"bert_{k}"]: r.span_log_probs(s, ["sun"]), [sentence], warmup=LATENCY_WARMUP, runs=LATENCY_RUNS)["mean_ms"] for k in ("fp32", "int8")}
    return {"fp32_mb": sizes["bert_fp32_only"], "int8_mb": sizes["bert_int8_only"]}, lat


def evaluate(worksheets: Path, emnist_root: Path = EMNIST_ROOT, save: bool = True) -> dict[str, Any]:
    torch.set_num_threads(THREADS)
    test_items = load_worksheet_items(worksheets, required=REQUIRED_ITEM_KEYS)
    check_splits(test_items, worksheets, expected_split(worksheets))
    warmup_items = load_worksheet_items(DEV_WORKSHEETS_PATH, required=REQUIRED_ITEM_KEYS)[:WARMUP_ITEMS]
    check_splits(warmup_items, DEV_WORKSHEETS_PATH, "val")
    params = load_fusion_params(PARAMS_PATH)
    models = load_models(params, save_int8=save)
    train_ds, test_ds = load_emnist_dataset(emnist_root, train=True), load_emnist_dataset(emnist_root, train=False)
    for item in warmup_items:
        run_item(item, item_to_strip(item, train_ds), models)
    strips = [item_to_strip(it, test_ds if it["emnist_split"] == "test" else train_ds) for it in test_items]
    ladder, records = word_ladder(test_items, strips, models)
    vision, vision_counts = vision_table(models, emnist_root)
    bert_sizes, bert_lat = bert_standalone(models)
    strip_char = ladder["raw_fp32"]["char_acc_pct"]
    gap = vision["char_cnn_fp32"]["acc_36_pct"] - strip_char
    note = f"gap {gap:+.2f} pp ({'INVESTIGATE: exceeds' if gap > bench_report.MAX_STRIP_GAP_PP else 'within'} {bench_report.MAX_STRIP_GAP_PP} pp)"
    recall = {k: 100.0 * sum(r[f"target_in_pool_{k}"] for r in records) / len(records) for k in ("fp32", "int8")}
    result = {"hardware": bench_report.hardware_metadata(), "worksheets": Path(worksheets).as_posix(), "vision": vision, "vision_n": vision_counts["n"], "ladder": ladder, "ladder_n": len(records),
              "recall_pct": recall, "strip_char_acc_pct": strip_char, "strip_gap_note": note, "bert_sizes": bert_sizes, "bert_latency_ms": bert_lat, "fusion_params": params}
    result["checks"] = bench_report.pass_fail_checks(vision, ladder, bert_sizes)
    predictions = {"worksheets": result["worksheets"], "rows": prediction_records(records), "vision": vision_counts}
    return {**result, "records": records, "predictions": predictions}


def save_outputs(result: Mapping[str, Any]) -> None:
    summary = {k: v for k, v in result.items() if k not in ("records", "predictions")}
    write_data(OUT["json"], summary, as_json=True)
    write_data(OUT["md"], bench_report.render_markdown(summary))
    write_data(OUT["failures"], bench_report.select_failures(result["records"]), as_json=True)
    write_data(OUT["predictions"], result["predictions"], as_json=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU benchmark of the vision models and the word-correction ladder.")
    parser.add_argument("--worksheets", type=Path, default=TEST_WORKSHEETS_PATH, help="Worksheet JSON to evaluate (debug on the dev file).")
    parser.add_argument("--no-save", action="store_true", help="Print the Markdown summary without writing any file.")
    parser.add_argument("--retime", action="store_true", help="Timing-only re-measure: outputs must match the committed predictions exactly.")
    parser.add_argument("--passes", type=int, default=3, help="Timing passes for --retime (the median pass is reported).")
    parser.add_argument("--superseded-file", type=Path, default=None, help="--retime: where to keep the timings being replaced.")
    parser.add_argument("--superseded-reason", default=None, help="--retime: why those timings are replaced.")
    args = parser.parse_args(argv)
    try:
        if args.retime:
            from scripts.retime_benchmark import retime

            extra = {k: v for k, v in (("superseded", args.superseded_file), ("reason", args.superseded_reason)) if v is not None}
            summary = retime(args.worksheets, args.passes, **extra)
            sys.stdout.write(bench_report.render_markdown(summary) + "\n")
            return 0
        result = evaluate(args.worksheets, save=not args.no_save)
        if not args.no_save:
            save_outputs(result)
    except OSError as exc:
        sys.stderr.write(describe_cli_os_error(exc) + "\n")
        return 1
    except (ValueError, TypeError, RuntimeError, KeyError, IndexError, ZeroDivisionError) as exc:
        sys.stderr.write(f"Error: {' '.join(str(exc).split()) or 'benchmark failed.'}\n")
        return 1
    sys.stdout.write(bench_report.render_markdown({k: v for k, v in result.items() if k not in ("records", "predictions")}) + "\n")
    sys.stdout.write(json.dumps(result["checks"]) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

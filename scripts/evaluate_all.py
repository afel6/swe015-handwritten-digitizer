"""End-to-end CPU benchmark: vision models (Table 1) and the word-correction ladder (Table 2).

Run it alone on the machine. All timing is on CPU with torch.set_num_threads(4) and time.perf_counter().
The test worksheets are read once: one pass gives both accuracy and per-item latency.
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

import torch

from data._io_errors import describe_os_error
from data.emnist_data import extract_oriented_tensors
from data.prepare_data import (
    DEV_WORKSHEETS_PATH,
    EMNIST_ROOT,
    TEST_WORKSHEETS_PATH,
    load_emnist_dataset,
    read_data,
    write_data,
)
from data.preprocess import (
    CANONICAL_36_CHARS,
    EMNIST_BALANCED_CHARS,
    fold_47_to_36_probs,
)
from data.word_pipeline import item_to_strip, strip_to_probs
from models.bert_reranker import BertMLMReranker
from models.char_models import load_char_cnn_fp32, load_dense_mlp
from models.classical_nlp import (
    ClassicalNLPCorrector,
    normalize_word,
)
from models.ladder import run_ladder, timed
from models.quantization import (
    benchmark_cpu_latency,
    get_model_disk_size_mb,
    load_quantized_cnn,
    save_quantized_state_dict,
)
from scripts import bench_report

THREADS, WARMUP_ITEMS, LATENCY_RUNS = 4, 10, 100
WEIGHTS = {"mlp": Path("weights/dense_mlp.pt"), "cnn_fp32": Path("weights/char_cnn_fp32.pt"), "cnn_int8": Path("weights/char_cnn_int8.pt"), "ngram": Path("weights/ngram_lm.pkl"), "bert_int8": Path("weights/bert_int8_dynamic.pt")}
OUT = {"json": Path("results/benchmark_results.json"), "md": Path("results/benchmark_results.md"), "failures": Path("results/failure_examples.json")}
PARAMS_PATH = Path("results/bert_fusion_params.json")
_FOLD = torch.tensor([CANONICAL_36_CHARS.index(ch.upper()) for ch in EMNIST_BALANCED_CHARS])
_LABELS = {"raw_fp32": "(1) Trained NN, no correction: FP32 CNN", "raw_int8": "(1b) parity: INT8 CNN, no correction", "classical": "(2) Classical: FP32 CNN + SymSpell + n-gram fusion",
           "bert_fp32": "(3) Transformer: FP32 CNN + same pool + FP32 BERT", "quantized_app": "(4) Quantized Transformer [App]: INT8 CNN + INT8 BERT", "ablation_fp32cnn_int8bert": "(ablation) FP32 CNN + same pool + INT8 BERT"}


def load_models(save_int8: bool = True) -> dict[str, Any]:
    """Load every artifact once; optionally save the INT8 BERT state_dict for the app (sized from that file)."""
    bert_fp32 = BertMLMReranker()
    bert_int8 = bert_fp32.to_int8()
    if save_int8 or not WEIGHTS["bert_int8"].is_file():
        save_quantized_state_dict(bert_int8.model, WEIGHTS["bert_int8"], metadata={"base_model": bert_fp32.model_name, "quantization": "dynamic_int8_linear"})
    return {"mlp": load_dense_mlp(WEIGHTS["mlp"]), "cnn_fp32": load_char_cnn_fp32(WEIGHTS["cnn_fp32"]), "cnn_int8": load_quantized_cnn(WEIGHTS["cnn_int8"]),
            "ngram": ClassicalNLPCorrector.load(WEIGHTS["ngram"]), "bert_fp32": bert_fp32, "bert_int8": bert_int8,
            "params": read_data(PARAMS_PATH, as_json=True)["params"]}


def vision_table(models: Mapping[str, Any], emnist_root: Path) -> tuple[dict[str, Any], int]:
    """Recompute EMNIST balanced test accuracy (47 and 36-folded), disk size and CPU latency per vision model."""
    images, labels = extract_oriented_tensors(load_emnist_dataset(emnist_root, train=False))
    table = {}
    for name, key in (("dense_mlp_fp32", "mlp"), ("char_cnn_fp32", "cnn_fp32"), ("char_cnn_int8", "cnn_int8")):
        model = models[key]
        with torch.inference_mode():
            probs = torch.cat([torch.softmax(model(chunk), dim=1) for chunk in images.split(2048)])
        acc47 = float((probs.argmax(1) == labels).float().mean()) * 100.0
        acc36 = float((fold_47_to_36_probs(probs).argmax(1) == _FOLD[labels]).float().mean()) * 100.0
        lat = {b: benchmark_cpu_latency(model, [images[i : i + b] for i in range(0, 110 * b, b)], warmup=10, runs=LATENCY_RUNS)["mean_ms"] for b in (1, 8)}
        table[name] = {"acc_47_pct": acc47, "acc_36_pct": acc36, "disk_mb": get_model_disk_size_mb(WEIGHTS[key]), "latency_b1_ms": lat[1], "latency_b8_ms": lat[8]}
    return table, int(labels.numel())


def run_item(item: Mapping[str, Any], strip: Any, models: Mapping[str, Any]) -> dict[str, Any]:
    """Run every ladder row on one strip (models.ladder.run_ladder); returns outputs plus per-row end-to-end latency (ms)."""
    sentence, n = str(item["sentence"]), len(str(item["target_word"]))
    p32, t32 = timed(lambda: strip_to_probs(strip, models["cnn_fp32"], num_boxes=n))
    p8, t8 = timed(lambda: strip_to_probs(strip, models["cnn_int8"], num_boxes=n))
    ladder = run_ladder(sentence, p32, p8, models, {"fp32": t32, "int8": t8})
    outputs = {k: row["predicted_word"] for k, row in ladder["rows"].items()}
    target = normalize_word(str(item["target_word"]))
    return {"id": item["id"], "sentence": sentence, "target": target, "outputs": outputs, "latency_ms": {k: row["latency_ms"] for k, row in ladder["rows"].items()},
            "raw_int8": outputs["raw_int8"], "target_in_pool_fp32": target in ladder["pools"]["fp32"], "target_in_pool_int8": target in ladder["pools"]["int8"],
            "numeric_preserved_int8": ladder["numeric_preserved_int8"]}


def row_sizes(models: Mapping[str, Any]) -> dict[str, float]:
    """Sum of every artifact each pipeline loads (MB)."""
    cnn32, cnn8, lm = (get_model_disk_size_mb(WEIGHTS[k]) for k in ("cnn_fp32", "cnn_int8", "ngram"))
    bert32, bert8 = get_model_disk_size_mb(models["bert_fp32"].model), get_model_disk_size_mb(WEIGHTS["bert_int8"])
    return {"raw_fp32": cnn32, "raw_int8": cnn8, "classical": cnn32 + lm, "bert_fp32": cnn32 + lm + bert32, "quantized_app": cnn8 + lm + bert8, "ablation_fp32cnn_int8bert": cnn32 + lm + bert8, "bert_fp32_only": bert32, "bert_int8_only": bert8}


def word_ladder(items: Sequence[Mapping[str, Any]], strips: Sequence[Any], models: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [run_item(it, st, models) for it, st in zip(items, strips, strict=True)]
    sizes, n_chars = row_sizes(models), sum(len(r["target"]) for r in records)
    ladder = {}
    for key, label in _LABELS.items():
        scores = [bench_report.word_scores(r["outputs"][key], r["target"]) for r in records]
        ladder[key] = {"label": label, **bench_report.summarize_row([s[0] for s in scores], [s[1] for s in scores], n_chars, [r["latency_ms"][key] for r in records], sizes[key])}
    return ladder, records


def bert_standalone(models: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    sizes = row_sizes(models)
    sentence = "The [MASK] is the closest star to the Earth."
    lat = {k: benchmark_cpu_latency(lambda s, r=models[f"bert_{k}"]: r.span_log_probs(s, ["sun"]), [sentence], warmup=10, runs=LATENCY_RUNS)["mean_ms"] for k in ("fp32", "int8")}
    return {"fp32_mb": sizes["bert_fp32_only"], "int8_mb": sizes["bert_int8_only"]}, lat


def evaluate(worksheets: Path, emnist_root: Path = EMNIST_ROOT, save: bool = True) -> dict[str, Any]:
    torch.set_num_threads(THREADS)
    test_items = read_data(worksheets, as_json=True)
    if not isinstance(test_items, list) or not test_items or any(not isinstance(it, dict) or "sentence" not in it or "target_word" not in it for it in test_items):
        raise ValueError(f"{worksheets} must hold a non-empty list of worksheet items with sentence and target_word.")
    models = load_models(save_int8=save)
    train_ds, test_ds = load_emnist_dataset(emnist_root, train=True), load_emnist_dataset(emnist_root, train=False)
    for item in read_data(DEV_WORKSHEETS_PATH, as_json=True)[:WARMUP_ITEMS]:
        run_item(item, item_to_strip(item, train_ds), models)
    strips = [item_to_strip(it, test_ds if it["emnist_split"] == "test" else train_ds) for it in test_items]
    ladder, records = word_ladder(test_items, strips, models)
    vision, vision_n = vision_table(models, emnist_root)
    bert_sizes, bert_lat = bert_standalone(models)
    strip_char = ladder["raw_fp32"]["char_acc_pct"]
    gap = vision["char_cnn_fp32"]["acc_36_pct"] - strip_char
    note = f"gap {gap:+.2f} pp ({'INVESTIGATE: exceeds' if gap > bench_report.MAX_STRIP_GAP_PP else 'within'} {bench_report.MAX_STRIP_GAP_PP} pp)"
    recall = {k: 100.0 * sum(r[f"target_in_pool_{k}"] for r in records) / len(records) for k in ("fp32", "int8")}
    result = {"hardware": bench_report.hardware_metadata(), "worksheets": os.fspath(worksheets), "vision": vision, "vision_n": vision_n, "ladder": ladder, "ladder_n": len(records),
              "recall_pct": recall, "strip_char_acc_pct": strip_char, "strip_gap_note": note, "bert_sizes": bert_sizes, "bert_latency_ms": bert_lat, "fusion_params": models["params"]}
    result["checks"] = bench_report.pass_fail_checks(vision, ladder, bert_sizes)
    return {**result, "records": records}


def save_outputs(result: Mapping[str, Any]) -> None:
    records = result["records"]
    summary = {k: v for k, v in result.items() if k != "records"}
    write_data(OUT["json"], summary, as_json=True)
    write_data(OUT["md"], bench_report.render_markdown(summary))
    write_data(OUT["failures"], bench_report.select_failures(records), as_json=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU benchmark of the vision models and the word-correction ladder.")
    parser.add_argument("--worksheets", type=Path, default=TEST_WORKSHEETS_PATH, help="Worksheet JSON to evaluate (debug on the dev file).")
    parser.add_argument("--no-save", action="store_true", help="Print the Markdown summary without writing result files.")
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.worksheets, save=not args.no_save)
        if not args.no_save:
            save_outputs(result)
    except OSError as exc:
        sys.stderr.write(f"File system error: {describe_os_error(exc)}\n")
        return 1
    except (ValueError, TypeError, RuntimeError, KeyError, IndexError, ZeroDivisionError) as exc:
        sys.stderr.write(f"Error: {' '.join(str(exc).splitlines()).strip() or 'benchmark failed.'}\n")
        return 1
    sys.stdout.write(bench_report.render_markdown({k: v for k, v in result.items() if k != "records"}) + "\n")
    sys.stdout.write(json.dumps(result["checks"]) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Qwen3-VL reference on the 300 test worksheets: an outside model, run zero-shot through local Ollama.

It is a reference, not part of the graded ladder: no tuning, no pass/fail check, GPU timing kept apart.
Two modes: the boxed strip alone (compare with the raw CNN row) and the strip plus the sentence
(compare with the BERT rows). Writes results/vlm_reference.json; recompute_metrics.py re-derives it.

    python scripts/benchmark_vlm.py            # both modes, all 300 items (about 15 minutes on an RTX 5070 Ti)
    python scripts/benchmark_vlm.py --limit 5  # quick smoke run to a scratch file
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import vlm_ocr  # noqa: E402
from scripts.recompute_metrics import normalize, row_metrics  # noqa: E402

MODES = ("strip_only", "strip_sentence")
LABEL = "Reference, not trained by us, GPU"
OUT = Path("results/vlm_reference.json")
_STRIP_TEXT = ("This image shows one handwritten word written in a row of boxes, one character per box; some boxes may be empty. "
               "Reply with only that word in capital letters, with no spaces, punctuation or explanation.")
_SENTENCE_TEXT = " The word fills the blank ([MASK]) in this sentence: {sentence}"

Ask = Callable[[bytes, str], Mapping[str, Any]]


def build_prompt(mode: str, sentence: str) -> str:
    if mode == "strip_only":
        return _STRIP_TEXT
    if mode == "strip_sentence":
        return _STRIP_TEXT + _SENTENCE_TEXT.format(sentence=sentence)
    raise ValueError(f"Unknown mode {mode!r}; use one of {MODES}.")


def run_mode(items: Sequence[Mapping[str, Any]], strips: Sequence[bytes], mode: str, ask: Ask) -> list[dict[str, Any]]:
    """One zero-shot call per worksheet; the reply is normalised exactly like the ladder's words."""
    records = []
    for item, strip in zip(items, strips, strict=True):
        reply = ask(strip, build_prompt(mode, str(item["sentence"])))
        text = str(reply.get("text") or "")
        records.append({"item_id": item["id"], "target_word": item["target_word"], "predicted_word": normalize(text),
                        "raw_reply": text, "latency_ms": float(reply["latency_ms"])})
    return records


def build_report(runs: Mapping[str, list[dict[str, Any]]], meta: Mapping[str, Any]) -> dict[str, Any]:
    out_runs = {}
    for mode, records in runs.items():
        metrics = row_metrics(records)
        out_runs[mode] = {"word_acc_pct": metrics["word_acc_pct"], "char_acc_pct": metrics["char_acc_pct"],
                          "latency_mean_ms": metrics["latency_mean_ms"], "records": records}
    sizes = {len(r) for r in runs.values()}
    return {
        "label": LABEL, "model": vlm_ocr.MODEL, "digest": meta.get("digest"), "ollama_version": meta.get("ollama_version"),
        "gpu": meta.get("gpu"), "n": sizes.pop() if len(sizes) == 1 else None,
        "settings": {"temperature": 0, "seed": vlm_ocr.SEED, "logprobs": True, "prompts": {m: build_prompt(m, "<sentence>") for m in MODES}},
        "runs": out_runs,
    }


def _get_json(path: str) -> Any:
    with urllib.request.urlopen(f"{vlm_ocr.OLLAMA_URL}{path}", timeout=vlm_ocr.STATUS_TIMEOUT_S) as resp:
        return json.load(resp)


def run_metadata() -> dict[str, Any]:
    digest = next((m.get("digest") for m in _get_json("/api/tags").get("models", []) if m.get("name") == vlm_ocr.MODEL), None)
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "unknown"
    return {"digest": digest, "ollama_version": _get_json("/api/version").get("version"), "gpu": gpu}


def load_test_strips(limit: int | None) -> tuple[list[dict[str, Any]], list[bytes]]:
    from data.prepare_data import EMNIST_ROOT, TEST_WORKSHEETS_PATH, load_emnist_dataset, load_worksheet_items
    from data.word_pipeline import item_to_strip
    from scripts import evaluate_all as ev

    items = load_worksheet_items(TEST_WORKSHEETS_PATH, required=ev.REQUIRED_ITEM_KEYS)[:limit]
    datasets = {"test": load_emnist_dataset(EMNIST_ROOT, train=False), "train": load_emnist_dataset(EMNIST_ROOT, train=True)}
    strips = []
    for item in items:
        buf = io.BytesIO()
        item_to_strip(item, datasets["test" if item["emnist_split"] == "test" else "train"]).save(buf, format="PNG")
        strips.append(buf.getvalue())
    return items, strips


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen3-VL reference on the test worksheets (needs local Ollama and the model).")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--limit", type=int, default=None, help="only the first N worksheets (smoke run)")
    parser.add_argument("--out", type=Path, default=None, help=f"output JSON (default {OUT}; a --limit run never writes there)")
    args = parser.parse_args(argv)
    out = args.out or (OUT if args.limit is None else Path(f"vlm_reference_smoke_{args.limit}.json"))
    status = vlm_ocr.status()
    if not status["model_ready"]:
        sys.stderr.write(f"Error: {status['detail']}\n")
        return 1
    try:
        items, strips = load_test_strips(args.limit)
        runs = {}
        for mode in args.modes:
            runs[mode] = run_mode(items, strips, mode, lambda img, prompt: vlm_ocr.transcribe(img, prompt))
            sys.stdout.write(f"{mode}: word acc {row_metrics(runs[mode])['word_acc_pct']:.2f} % on {len(items)} items\n")
        report = build_report(runs, run_metadata())
    except (OSError, ValueError, KeyError, vlm_ocr.OcrUnavailable) as exc:
        sys.stderr.write(f"Error: {' '.join(str(exc).split())}\n")
        return 1
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sys.stdout.write(f"Wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

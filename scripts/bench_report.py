"""Pure helpers for the benchmark: hardware metadata, word metrics, PASS/FAIL checks, failures, Markdown."""

from __future__ import annotations

import ctypes
import importlib.metadata
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from models.classical_nlp import CANDIDATE_POOL_SIZE, normalize_word

MAX_INT8_ACC_DROP_PP = 1.5
MIN_BERT_SIZE_REDUCTION = 2.0
MAX_STRIP_GAP_PP = 5.0
_BYTES_PER_GB = 1024.0**3
_PCT = 100.0


def _cpu_model() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            return platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            return next((ln.split(":", 1)[1].strip() for ln in handle if ln.startswith("model name")), platform.processor() or "unknown")
    except OSError:
        return platform.processor() or "unknown"


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in ("total", "avail", "page_total", "page_avail", "virt_total", "virt_avail", "ext")]


def _windows_total_ram_gb(kernel32: Any) -> float | None:
    """Total RAM via GlobalMemoryStatusEx; None if the call fails (never a misleading 0)."""
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) or status.total == 0:
        return None
    return round(status.total / _BYTES_PER_GB, 1)


def _ram_gb() -> float | None:
    if sys.platform == "win32":
        return _windows_total_ram_gb(ctypes.windll.kernel32)
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / _BYTES_PER_GB, 1)
    except (ValueError, OSError, AttributeError):
        return None


def hardware_metadata() -> dict[str, Any]:
    """Machine and library facts captured at run time (the GPU is listed but never used for timing)."""
    return {
        "cpu_model": _cpu_model(), "logical_cores": os.cpu_count(), "ram_gb": _ram_gb(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"), "torch_threads": torch.get_num_threads(), "timing_device": "cpu",
    }


def word_scores(prediction: str, target: str) -> tuple[bool, int]:
    """Case-insensitive word hit and the number of position-matching characters."""
    pred, tgt = normalize_word(prediction), normalize_word(target)
    return pred == tgt, sum(a == b for a, b in zip(pred, tgt))


def summarize_row(hits: Sequence[bool], char_hits: Sequence[int], n_chars: int, latencies_ms: Sequence[float], size_mb: float) -> dict[str, Any]:
    """Aggregate one ladder row: word/char accuracy (%), artifact size and mean end-to-end latency."""
    return {
        "word_acc_pct": _PCT * sum(hits) / len(hits), "char_acc_pct": _PCT * sum(char_hits) / n_chars,
        "size_mb": float(size_mb), "latency_mean_ms": float(sum(latencies_ms) / len(latencies_ms)), "n": len(hits),
    }


def pass_fail_checks(vision: Mapping[str, Any], ladder: Mapping[str, Any], bert_sizes: Mapping[str, float]) -> list[dict[str, Any]]:
    """Report-only checks; a FAIL is a measured result, never a reason to retune on test."""
    raw, ngram, bert, app = (ladder[k]["word_acc_pct"] for k in ("raw_fp32", "classical", "bert_fp32", "quantized_app"))
    reduction = bert_sizes["fp32_mb"] / bert_sizes["int8_mb"]
    bert_drop = ladder["bert_fp32"]["word_acc_pct"] - ladder["ablation_fp32cnn_int8bert"]["word_acc_pct"]
    cnn_drop = vision["char_cnn_fp32"]["acc_36_pct"] - vision["char_cnn_int8"]["acc_36_pct"]
    checks = [
        ("Raw < N-gram < BERT FP32 (word acc)", raw < ngram < bert, f"{raw:.2f} / {ngram:.2f} / {bert:.2f} %"),
        ("Raw < N-gram < quantized app (word acc)", raw < ngram < app, f"{raw:.2f} / {ngram:.2f} / {app:.2f} %"),
        (f"INT8 BERT size reduction >= {MIN_BERT_SIZE_REDUCTION}x", reduction >= MIN_BERT_SIZE_REDUCTION, f"{reduction:.2f}x"),
        (f"INT8 BERT word-acc drop <= {MAX_INT8_ACC_DROP_PP} pp", bert_drop <= MAX_INT8_ACC_DROP_PP, f"{bert_drop:+.2f} pp"),
        (f"INT8 CNN 36-class acc drop <= {MAX_INT8_ACC_DROP_PP} pp", cnn_drop <= MAX_INT8_ACC_DROP_PP, f"{cnn_drop:+.2f} pp"),
    ]
    return [{"check": name, "result": "PASS" if ok else "FAIL", "value": value} for name, ok, value in checks]


def select_failures(records: Sequence[Mapping[str, Any]], minimum: int = 3, limit: int = 8) -> list[dict[str, Any]]:
    """Pick real failures of the app model, most informative first (re-ranker errors, then pool misses)."""
    failed = [r for r in records if normalize_word(r["outputs"]["quantized_app"]) != normalize_word(r["target"])]
    failed.sort(key=lambda r: (not r["target_in_pool_int8"], r["id"]))
    picked = [{**r, "why": explain_failure(r)} for r in failed[:limit]]
    if len(picked) < minimum:
        others = [r for r in records if r not in failed and any(normalize_word(v) != normalize_word(r["target"]) for v in r["outputs"].values())]
        picked += [{**r, "why": explain_failure(r)} for r in others[: minimum - len(picked)]]
    return picked


def explain_failure(record: Mapping[str, Any]) -> str:
    """Plain-language reason for a failure, from the recorded pipeline state."""
    target, raw = normalize_word(record["target"]), normalize_word(record["raw_int8"])
    if record.get("numeric_preserved_int8"):
        return f"numeric guard kept the confident CNN digits {raw!r}"
    if not record["target_in_pool_int8"] and target.isdigit():
        return (f"CNN read {raw!r} with letters among the digits; the SymSpell lexicon holds words only, and a digit string enters the pool only as the raw read "
                f"when every box reads as a digit, so {target!r} could not be a candidate")
    if not record["target_in_pool_int8"]:
        return f"CNN read {raw!r}; {target!r} is not among the {CANDIDATE_POOL_SIZE} pool candidates (lexicon words within edit distance 2 of the lattice reads)"
    return f"{target!r} was in the candidate pool, but the fused visual + BERT + edit score ranked {normalize_word(record['outputs']['quantized_app'])!r} higher"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *("| " + " | ".join(r) + " |" for r in rows)]


def render_markdown(result: Mapping[str, Any]) -> str:
    """Both tables, hardware, candidate recall and checks as Markdown."""
    vis_rows = [[name, f"{v['acc_47_pct']:.2f}", f"{v['acc_36_pct']:.2f}", f"{v['disk_mb']:.2f}", f"{v['latency_b1_ms']:.3f}", f"{v['latency_b8_ms']:.3f}"] for name, v in result["vision"].items()]
    lad_rows = [[v["label"], f"{v['word_acc_pct']:.2f}", f"{v['char_acc_pct']:.2f}", f"{v['size_mb']:.1f}", f"{v['latency_mean_ms']:.1f}"] for v in result["ladder"].values()]
    hw = result["hardware"]
    lines = ["# Benchmark results", "", f"CPU timing only: {hw['cpu_model']}, {hw['torch_threads']} torch threads, {hw['logical_cores']} logical cores, {'unknown' if hw['ram_gb'] is None else hw['ram_gb']} GB RAM; GPU present but unused for timing: {hw['gpu']}.",
             f"{hw['os']}, Python {hw['python']}, torch {hw['torch']}, transformers {hw['transformers']}.", "",
             f"## Table 1 - Vision (EMNIST balanced test, N={result['vision_n']})", "",
             *_table(["Model", "47-class acc %", "36-folded acc %", "Disk MB", "CPU b1 ms", "CPU b8 ms"], vis_rows), "",
             f"## Table 2 - Word ladder (N={result['ladder_n']} worksheets from {Path(str(result.get('worksheets', '?'))).name}, boxed strips)", "",
             *_table(["Pipeline", "Word acc %", "Char acc %", "Size MB", "Mean ms/word"], lad_rows), "",
             f"Candidate recall (target in the shared pool): FP32 CNN {result['recall_pct']['fp32']:.2f} %, INT8 CNN {result['recall_pct']['int8']:.2f} % (ceiling for rows 2-4).",
             f"Strip-path char accuracy {result['strip_char_acc_pct']:.2f} % vs raw EMNIST 36-folded {result['vision']['char_cnn_fp32']['acc_36_pct']:.2f} %: {result['strip_gap_note']}",
             "Sizes sum the weight files each pipeline loads (FP32 BERT: serialized state_dict; INT8 BERT: saved state_dict file; the ~0.2 MB tokenizer vocab is not counted). Row (4) is the app pipeline measured offline: its pool uses the same fixed rule on the INT8 CNN's own read, so it can differ from the FP32-CNN pool of rows 2, 3 and the ablation.",
             f"Standalone BERT: FP32 {result['bert_sizes']['fp32_mb']:.1f} MB, {result['bert_latency_ms']['fp32']:.1f} ms; INT8 {result['bert_sizes']['int8_mb']:.1f} MB, {result['bert_latency_ms']['int8']:.1f} ms (one masked sentence, batch 1).", "",
             "## Checks (reported, not enforced)", "", *_table(["Check", "Result", "Value"], [[c["check"], c["result"], c["value"]] for c in result["checks"]]), ""]
    return "\n".join(lines)

"""Required model/result files and the fix command; no heavy imports, so it can run before torch loads."""

from __future__ import annotations

from pathlib import Path

WEIGHT_FILES = ("dense_mlp.pt", "char_cnn_fp32.pt", "char_cnn_int8.pt", "ngram_lm.pkl")
RESULT_FILES = ("bert_fusion_params.json", "ngram_metrics.json")
BERT_INT8_FILE = "bert_int8_dynamic.pt"
FIX_COMMAND = "python scripts/download_assets.py"

__all__ = ["BERT_INT8_FILE", "FIX_COMMAND", "RESULT_FILES", "WEIGHT_FILES", "missing_assets", "missing_assets_message"]


def missing_assets(weights_dir: Path, results_dir: Path) -> list[Path]:
    """Required files that are not present (the INT8 BERT file is optional: it is rebuilt and cached)."""
    wanted = [Path(weights_dir) / n for n in WEIGHT_FILES] + [Path(results_dir) / n for n in RESULT_FILES]
    return [p for p in wanted if not p.is_file()]


def missing_assets_message(missing: list[Path]) -> str:
    """One friendly line naming the first missing file and the exact command that fetches it."""
    return f"Missing model file: {missing[0]}. Run: {FIX_COMMAND}"

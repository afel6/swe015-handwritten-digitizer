"""The word-correction ladder shared by the benchmark (scripts/evaluate_all.py) and the app.

Given the folded (n_boxes, 36) probabilities of the FP32 and INT8 character CNNs for one word,
run every pipeline row on the SAME candidate pool and time each stage on the calling thread:
raw CNN reads, classical n-gram fusion, FP32 BERT, the quantized app model (INT8 CNN pool +
INT8 BERT) and the FP32-CNN + INT8-BERT ablation.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from data.preprocess import CANONICAL_36_CHARS
from models.classical_nlp import CANDIDATE_POOL_SIZE

ROW_KEYS = ("raw_fp32", "raw_int8", "classical", "bert_fp32", "quantized_app", "ablation_fp32cnn_int8bert")
_MS = 1000.0

__all__ = ["ROW_KEYS", "raw_word", "run_ladder", "timed"]


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run fn once; return its result and the wall time in milliseconds (time.perf_counter)."""
    start = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - start) * _MS


def raw_word(probs: np.ndarray) -> str:
    """Argmax read of a (n, 36) probability matrix."""
    return "".join(CANONICAL_36_CHARS[int(i)] for i in np.asarray(probs).argmax(axis=1))


def _raw_row(probs: np.ndarray, ms: float) -> dict[str, Any]:
    word = raw_word(probs)
    return {"predicted_word": word, "raw_word": word, "top_candidates": [], "latency_ms": ms}


def _ranked_row(result: Mapping[str, Any], ms: float) -> dict[str, Any]:
    top = [{k: v for k, v in c.items() if k in ("word", "score", "vis_log_prob", "weighted_edit_distance", "lm_log_prob", "bert_pll_norm")} for c in result["top_candidates"][:5]]
    return {"predicted_word": result["predicted_word"], "raw_word": result["raw_word"], "top_candidates": top, "latency_ms": ms}


def run_ladder(sentence: str, p32: np.ndarray, p8: np.ndarray, models: Mapping[str, Any], cnn_ms: Mapping[str, float]) -> dict[str, Any]:
    """Every ladder row for one word. Rows 2, 3 and the ablation rank the identical FP32-CNN pool."""
    lm, params = models["ngram"], models["params"]
    (pool32, meta32), tp32 = timed(lambda: lm.generate_candidates(p32, max_candidates=CANDIDATE_POOL_SIZE))
    (pool8, meta8), tp8 = timed(lambda: lm.generate_candidates(p8, max_candidates=CANDIDATE_POOL_SIZE))
    kw32 = {"params": params, "raw_word": meta32["raw_word"], "numeric_preserved": meta32["numeric_preserved"]}
    kw8 = {"params": params, "raw_word": meta8["raw_word"], "numeric_preserved": meta8["numeric_preserved"]}
    classical, tc = timed(lambda: lm.rank_pool(sentence, pool32, meta32))
    bert32, tb32 = timed(lambda: models["bert_fp32"].rerank_candidates(sentence, pool32, **kw32))
    ablation, tab = timed(lambda: models["bert_int8"].rerank_candidates(sentence, pool32, **kw32))
    app, tapp = timed(lambda: models["bert_int8"].rerank_candidates(sentence, pool8, **kw8))
    t32, t8 = cnn_ms["fp32"], cnn_ms["int8"]
    rows = {
        "raw_fp32": _raw_row(p32, t32), "raw_int8": _raw_row(p8, t8),
        "classical": _ranked_row(classical, t32 + tp32 + tc), "bert_fp32": _ranked_row(bert32, t32 + tp32 + tb32),
        "quantized_app": _ranked_row(app, t8 + tp8 + tapp), "ablation_fp32cnn_int8bert": _ranked_row(ablation, t32 + tp32 + tab),
    }
    pools = {"fp32": {c["word"] for c in pool32}, "int8": {c["word"] for c in pool8}}
    return {"rows": rows, "pools": pools, "numeric_preserved_int8": bool(meta8["numeric_preserved"])}

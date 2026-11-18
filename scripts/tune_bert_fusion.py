"""Tune the BERT re-ranker fusion weights on the dev worksheets only (never the 300-item test set).

Per dev item the shared candidate pool (n-gram SymSpell lattice over strip-path FP32 CNN
probabilities) and every candidate's WordPiece log-probabilities are computed ONCE; alpha,
lambda_vis, lambda_bert and gamma are then grid-searched offline with the continuous objective
and smallest-weights tie-break documented in models.fusion_objective.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

if str(_ROOT := Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from data._io_errors import describe_cli_os_error
from data.prepare_data import DEV_WORKSHEETS_PATH, load_worksheet_items, write_data
from models.bert_reranker import BertMLMReranker, fused_score
from models.classical_nlp import (
    CANDIDATE_POOL_SIZE,
    ClassicalNLPCorrector,
    normalize_word,
)
from models.fusion_objective import mean_target_log_softmax, search, simplex_points

PARAMS_PATH = Path("results/bert_fusion_params.json")
BERT_ALPHAS: tuple[float, ...] = (0.0, 0.5, 1.0)
FUSED_WEIGHTS = ("lambda_vis", "lambda_bert", "gamma")
SIMPLEX_STEP = 0.05


def build_cache(items: Sequence[Mapping[str, Any]], probs_list: Sequence[np.ndarray], corrector: ClassicalNLPCorrector, reranker: BertMLMReranker) -> list[dict[str, Any]]:
    """One record per dev item: target, numeric-guard flag, and per-candidate (vis, wed, piece log-probs)."""
    cache = []
    for item, probs in zip(items, probs_list, strict=True):
        cands, meta = corrector.generate_candidates(probs, max_candidates=CANDIDATE_POOL_SIZE)
        span = reranker.span_log_probs(str(item["sentence"]), [str(c["word"]) for c in cands])
        rows = [(str(c["word"]), float(c["vis_log_prob"]), float(c["weighted_edit_distance"]), span[str(c["word"])]) for c in cands]
        cache.append({"target": normalize_word(str(item["target_word"])), "numeric_preserved": bool(meta["numeric_preserved"]), "raw_word": str(meta["raw_word"]), "rows": rows})
    return cache


def bert_objective(cache: Sequence[Mapping[str, Any]], params: Mapping[str, float]) -> float:
    """Mean target log-softmax over items the weights can influence (target in pool, guard not engaged)."""
    scored = []
    for rec in cache:
        words = [r[0] for r in rec["rows"]]
        if rec["numeric_preserved"] or rec["target"] not in words:
            continue
        scores = np.asarray([fused_score(vis, lps, wed, params) for _, vis, wed, lps in rec["rows"]])
        scored.append((scores, words.index(rec["target"])))
    return mean_target_log_softmax(scored)


def cached_accuracy(cache: Sequence[Mapping[str, Any]], params: Mapping[str, float]) -> float:
    """Word accuracy of the fused ranking on the cached pools (numeric guard applied)."""
    hits = 0
    for rec in cache:
        if rec["numeric_preserved"]:
            hits += rec["raw_word"] == rec["target"]
            continue
        best = min(rec["rows"], key=lambda r: (-fused_score(r[1], r[3], r[2], params), r[0]))  # same tie-break as rerank_candidates
        hits += best[0] == rec["target"]
    return hits / len(cache) if cache else 0.0


def candidate_params() -> list[dict[str, float]]:
    """alpha on a plain grid x fused weights on the simplex (see models.fusion_objective)."""
    simplex = simplex_points(FUSED_WEIGHTS, step=SIMPLEX_STEP, minimum=SIMPLEX_STEP)
    return [{"alpha": float(a), **w} for a in sorted(BERT_ALPHAS) for w in simplex]


def grid_edges(params: Mapping[str, float]) -> list[str]:
    """Parameters that landed on the boundary of their searched range (reported, not hidden)."""
    cands = candidate_params()
    return [n for n in params if params[n] in (min(c[n] for c in cands), max(c[n] for c in cands))]


def tune(cache: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], float]:
    return search(candidate_params(), lambda p: bert_objective(cache, p))


def run(out_path: Path = PARAMS_PATH, cnn_weights: Path = Path("weights/char_cnn_fp32.pt")) -> dict[str, Any]:
    from scripts.train_ngram import compute_dev_cnn_probs

    dev_items = load_worksheet_items(DEV_WORKSHEETS_PATH)
    probs = compute_dev_cnn_probs(dev_items, cnn_weights=cnn_weights)
    corrector = ClassicalNLPCorrector.load()
    fp32 = BertMLMReranker()
    caches = {"fp32": build_cache(dev_items, probs, corrector, fp32), "int8": build_cache(dev_items, probs, corrector, fp32.to_int8())}
    tuned = {name: tune(cache) for name, cache in caches.items()}
    params, value = tuned["fp32"]
    in_pool = sum(rec["target"] in [r[0] for r in rec["rows"]] for rec in caches["fp32"])
    result = {
        "tuned_on": DEV_WORKSHEETS_PATH.as_posix(),
        "objective": "mean log-softmax of the target's fused score over the candidate pool (dev items with target in pool, numeric guard not engaged)",
        "tie_break": "candidates visited in ascending lexicographic order; only a strictly better finite objective replaces the best",
        "scale": "fused weights sum to 1 (simplex), because log-softmax otherwise always prefers larger weights",
        "search_space": {"alpha": list(BERT_ALPHAS), "simplex": list(FUSED_WEIGHTS), "simplex_step": SIMPLEX_STEP, "simplex_min": SIMPLEX_STEP}, "pool_size": CANDIDATE_POOL_SIZE,
        "params_on_grid_edge": grid_edges(tuned["fp32"][0]),
        "dev_items": len(dev_items), "dev_target_in_pool": in_pool,
        "params": params, "objective_value_fp32": value, "dev_word_acc_tuned_fp32": cached_accuracy(caches["fp32"], params),
        "int8_alone_params": tuned["int8"][0], "objective_value_int8_alone": tuned["int8"][1],
        "int8_alone_would_differ": tuned["int8"][0] != params, "objective_value_int8_with_fp32_params": bert_objective(caches["int8"], params),
    }
    write_data(Path(out_path), result, as_json=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune BERT re-ranker fusion weights on the dev worksheets.")
    parser.add_argument("--out", type=Path, default=PARAMS_PATH, help="Output JSON path.")
    args = parser.parse_args(argv)
    try:
        result = run(args.out)
    except OSError as exc:
        sys.stderr.write(describe_cli_os_error(exc) + "\n")
        return 1
    except (ValueError, TypeError, RuntimeError, KeyError, IndexError) as exc:
        sys.stderr.write(f"Error: {' '.join(str(exc).splitlines()).strip() or 'tuning failed.'}\n")
        return 1
    sys.stdout.write(f"BERT fusion tuned | params {result['params']} | objective {result['objective_value_fp32']:.4f} | dev acc {result['dev_word_acc_tuned_fp32']:.2%}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

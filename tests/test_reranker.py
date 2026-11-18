"""Tests for the BERT re-ranker, the shared cloze/strip/objective helpers, BERT tuning and the benchmark helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.ao.nn.quantized.dynamic import Linear as DynamicQuantizedLinear
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

from data.preprocess import CANONICAL_36_CHARS
from models import bert_reranker, classical_nlp
from models.bert_reranker import (
    BertMLMReranker,
    fused_score,
    normalize_masked_sentence,
    pll_norm,
)
from models.fusion_objective import grid_search, mean_target_log_softmax
from scripts import bench_report, evaluate_all
from scripts import tune_bert_fusion as tune_bert

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_PATH = REPO_ROOT / "data" / "worksheets_dev_50.json"
EMNIST_RAW = REPO_ROOT / "data" / "emnist" / "EMNIST" / "raw"
CNN_WEIGHTS = REPO_ROOT / "weights" / "char_cnn_fp32.pt"
needs_emnist_cnn = pytest.mark.skipif(not (EMNIST_RAW / "emnist-balanced-train-labels-idx1-ubyte").is_file() or not CNN_WEIGHTS.is_file(), reason="EMNIST or CNN weights missing")
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "sun", "is", "hot", "moon", "cold", "sci", "##ence", "s", "##il", "##en", "1945", "war", "in", "ended", "."]
_C2I = {ch: i for i, ch in enumerate(CANONICAL_36_CHARS)}


@pytest.fixture(scope="module")
def tiny_reranker() -> BertMLMReranker:
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, max_position_embeddings=64)
    tokenizer = BertTokenizer(vocab={w: i for i, w in enumerate(VOCAB)})
    return BertMLMReranker("tiny", model=BertForMaskedLM(config).eval(), tokenizer=tokenizer)


def _cand(word: str, vis: float = -1.0, wed: float = 0.0) -> dict[str, Any]:
    return {"word": word, "vis_log_prob": vis, "weighted_edit_distance": wed}


# ---------------------------------------------------------------- cloze normalisation


@pytest.mark.parametrize(("sentence", "expected"), [("The Sun ___ hot.", "The Sun [MASK] hot."), ("The Sun [Mask] hot.", "The Sun [MASK] hot."),
                                                   ("the sun [mask] hot", "the sun [MASK] hot"), ("Keep CASE [MASK] here", "Keep CASE [MASK] here")])
def test_normalize_rewrites_blanks_and_keeps_the_rest(sentence: str, expected: str) -> None:
    assert normalize_masked_sentence(sentence) == expected


@pytest.mark.parametrize("sentence", ["The __ is hot.", "No blank.", "Two [MASK] and ___ here.", "[mask] [MASK]"])
def test_normalize_requires_exactly_one_blank(sentence: str) -> None:
    with pytest.raises(ValueError):
        normalize_masked_sentence(sentence)


def test_normalize_length_limit_and_type() -> None:
    ok = "A [MASK] " + "x" * 491
    assert len(ok) == 500 and normalize_masked_sentence(ok) == ok
    with pytest.raises(ValueError):
        normalize_masked_sentence(ok + "x")
    with pytest.raises(TypeError):
        normalize_masked_sentence(None)  # type: ignore[arg-type]


def test_one_cloze_implementation_is_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    assert classical_nlp.normalize_masked_sentence is bert_reranker.normalize_masked_sentence

    def sentinel(_s: str) -> str:
        raise LookupError("shared normaliser called")

    monkeypatch.setattr(classical_nlp, "normalize_masked_sentence", sentinel)
    with pytest.raises(LookupError, match="shared normaliser"):
        classical_nlp.extract_cloze_context("The [MASK] is hot.")


# ---------------------------------------------------------------- scoring math and forward-pass contract


def test_single_piece_candidates_use_exactly_one_forward_pass(tiny_reranker: BertMLMReranker) -> None:
    before = tiny_reranker.forward_calls
    out = tiny_reranker.span_log_probs("the [MASK] is hot", ["sun", "moon", "cold", "hot"])
    assert tiny_reranker.forward_calls - before == 1
    assert all(len(v) == 1 for v in out.values())


def test_multi_piece_candidates_use_exactly_one_batched_pass(tiny_reranker: BertMLMReranker) -> None:
    before = tiny_reranker.forward_calls
    out = tiny_reranker.span_log_probs("the [MASK] is hot", ["science", "silence"])
    assert tiny_reranker.forward_calls - before == 1
    assert len(out["science"]) == 2 and len(out["silence"]) == 3
    before = tiny_reranker.forward_calls
    tiny_reranker.span_log_probs("the [MASK] is hot", ["sun", "science", "silence"])
    assert tiny_reranker.forward_calls - before == 2


def _hand_piece_log_prob(rr: BertMLMReranker, prefix: list[int], suffix: list[int], pieces: list[int], j: int) -> float:
    tok = rr.tokenizer
    seq = [tok.cls_token_id, *prefix, *pieces[:j], tok.mask_token_id, *pieces[j + 1 :], *suffix, tok.sep_token_id]
    with torch.no_grad():
        logits = rr.model(input_ids=torch.tensor([seq])).logits[0, len(prefix) + 1 + j]
    return float(torch.log_softmax(logits, dim=-1)[pieces[j]])


def test_batched_span_pll_matches_unbatched_hand_computation(tiny_reranker: BertMLMReranker) -> None:
    ids = {w: VOCAB.index(w) for w in VOCAB}
    prefix, suffix = [ids["the"]], [ids["is"], ids["hot"]]
    out = tiny_reranker.span_log_probs("The [MASK] is hot", ["silence", "science", "sun"])  # padding: 3-, 2- and 1-piece words
    for word, pieces in (("silence", [ids["s"], ids["##il"], ids["##ence"]]), ("science", [ids["sci"], ids["##ence"]])):
        expected = [_hand_piece_log_prob(tiny_reranker, prefix, suffix, pieces, j) for j in range(len(pieces))]
        assert out[word] == pytest.approx(expected, abs=1e-5)
    assert out["sun"] == pytest.approx([_hand_piece_log_prob(tiny_reranker, prefix, suffix, [ids["sun"]], 0)], abs=1e-5)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_pll_norm_divides_by_k_to_the_alpha(alpha: float) -> None:
    lps = [-1.0, -2.0, -3.5]
    assert pll_norm(lps, alpha) == pytest.approx(-6.5 / 3**alpha)
    with pytest.raises(ValueError):
        pll_norm([], alpha)


def test_alpha_rescales_three_piece_candidate_score(tiny_reranker: BertMLMReranker) -> None:
    rows = {a: tiny_reranker.rerank_candidates("the [MASK] is hot", [_cand("silence"), _cand("sun")], params={"alpha": a, "lambda_vis": 0.0, "lambda_bert": 1.0, "gamma": 0.0})["top_candidates"] for a in (0.0, 1.0)}
    norm = {a: next(r["bert_pll_norm"] for r in rows[a] if r["word"] == "silence") for a in rows}
    assert norm[0.0] == pytest.approx(3.0 * norm[1.0])


def test_fused_score_formula() -> None:
    params = {"alpha": 1.0, "lambda_vis": 2.0, "lambda_bert": 0.5, "gamma": 3.0}
    assert fused_score(-1.0, [-2.0, -4.0], 0.5, params) == pytest.approx(2.0 * -1.0 + 0.5 * (-6.0 / 2) - 3.0 * 0.5)


def test_rerank_prefers_fused_best_and_reports_components(tiny_reranker: BertMLMReranker) -> None:
    out = tiny_reranker.rerank_candidates("the [MASK] is hot", [_cand("moon", vis=-9.0), _cand("sun", vis=-0.1)], params={"alpha": 1.0, "lambda_vis": 5.0, "lambda_bert": 0.1, "gamma": 1.0})
    assert out["predicted_word"] == "sun"
    assert {"bert_pll_norm", "bert_piece_log_probs", "bert_pieces", "score", "vis_log_prob"} <= set(out["top_candidates"][0])
    with pytest.raises(ValueError):
        tiny_reranker.rerank_candidates("the [MASK] is hot", [])
    with pytest.raises(ValueError):
        tiny_reranker.rerank_candidates("the [MASK] is hot", [_cand("sun")], params={"alpha": 1.0})


def test_numeric_guard_keeps_raw_digits(tiny_reranker: BertMLMReranker) -> None:
    pool = [_cand("1945", vis=-20.0, wed=0.0), _cand("the", vis=0.0)]
    params = {"alpha": 1.0, "lambda_vis": 1.0, "lambda_bert": 1.0, "gamma": 1.0}
    assert tiny_reranker.rerank_candidates("the war ended in [MASK]", pool, params=params, raw_word="1945", numeric_preserved=True)["predicted_word"] == "1945"
    assert tiny_reranker.rerank_candidates("the war ended in [MASK]", pool, params=params, raw_word="1945", numeric_preserved=False)["predicted_word"] == "the"


def test_to_int8_returns_new_quantized_reranker_and_leaves_fp32_untouched(tiny_reranker: BertMLMReranker) -> None:
    before = {k: v.clone() for k, v in tiny_reranker.model.state_dict().items()}
    int8 = tiny_reranker.to_int8()
    assert int8 is not tiny_reranker and int8.model is not tiny_reranker.model
    assert any(isinstance(m, DynamicQuantizedLinear) for m in int8.model.modules())
    assert not any(isinstance(m, DynamicQuantizedLinear) for m in tiny_reranker.model.modules())
    assert any(type(m) is nn.Linear for m in tiny_reranker.model.modules())
    after = tiny_reranker.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert set(int8.span_log_probs("the [MASK] is hot", ["sun"])) == {"sun"}


def _real_bert() -> BertMLMReranker:
    from transformers import AutoTokenizer

    try:  # never download in tests: only run when the model is already cached
        AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
        model = BertForMaskedLM.from_pretrained("bert-base-uncased", local_files_only=True)
    except OSError:
        pytest.skip("bert-base-uncased is not in the local Hugging Face cache")
    return BertMLMReranker(model=model)


def test_real_bert_int8_is_at_least_twice_smaller_and_still_ranks_sensibly() -> None:
    from models.quantization import get_model_disk_size_mb

    fp32 = _real_bert()
    int8 = fp32.to_int8()
    assert get_model_disk_size_mb(fp32.model) / get_model_disk_size_mb(int8.model) >= 2.0
    params = {"alpha": 1.0, "lambda_vis": 0.0, "lambda_bert": 1.0, "gamma": 0.0}
    for rr in (fp32, int8):
        out = rr.rerank_candidates("The [MASK] is the closest star to the Earth.", [_cand("sun"), _cand("moon"), _cand("table")], params=params)
        assert out["predicted_word"] == "sun"  # single-piece words: multi-piece span PLL is inflated (visible sibling pieces)


# ---------------------------------------------------------------- objective and tuning


def test_mean_target_log_softmax_matches_hand_value() -> None:
    items = [(np.array([2.0, 0.0]), 0), (np.array([1.0, 1.0, 1.0]), 2)]
    expected = ((2.0 - math.log(math.exp(2) + 1)) + math.log(1 / 3)) / 2
    assert mean_target_log_softmax(items) == pytest.approx(expected)
    assert mean_target_log_softmax([]) == -math.inf
    with pytest.raises(ValueError):
        mean_target_log_softmax([(np.array([1.0]), 3)])


def test_simplex_points_sum_to_one_and_respect_minimum() -> None:
    from models.fusion_objective import simplex_points

    pts = simplex_points(("a", "b", "c"), step=0.25, minimum=0.25)
    assert pts == [{"a": 0.25, "b": 0.25, "c": 0.5}, {"a": 0.25, "b": 0.5, "c": 0.25}, {"a": 0.5, "b": 0.25, "c": 0.25}]
    with pytest.raises(ValueError):
        simplex_points(("a", "b"), minimum=0.6)


def test_simplex_search_is_scale_free_unlike_a_free_grid() -> None:
    from models.fusion_objective import search, simplex_points

    items = [(np.array([3.0, 1.0]), 0), (np.array([2.0, 0.0]), 0)]  # features: (vis, lm) per candidate
    feats = [np.array([[1.0, 2.0], [0.0, 1.0]]), np.array([[1.0, 1.0], [0.0, 0.0]])]
    objective = lambda w: mean_target_log_softmax([(f @ np.array([w["a"], w["b"]]), i) for f, (_, i) in zip(feats, items, strict=True)])  # noqa: E731
    free, _ = grid_search({"a": (0.5, 1.0, 2.0), "b": (0.5, 1.0, 2.0)}, objective)
    assert free == {"a": 2.0, "b": 2.0}  # the degenerate free-grid optimum: largest everything
    best, _ = search(simplex_points(("a", "b"), step=0.1, minimum=0.1), objective)
    assert sum(best.values()) == pytest.approx(1.0)


def test_search_skips_non_finite_objectives() -> None:
    from models.fusion_objective import search

    best, value = search([{"a": 0.0}, {"a": 1.0}], lambda p: math.nan if p["a"] == 0.0 else -1.0)
    assert best == {"a": 1.0} and value == -1.0
    with pytest.raises(ValueError):
        search([{"a": 0.0}], lambda p: math.nan)


@pytest.mark.parametrize(("step", "minimum"), [(0.3, 0.3), (0.25, 0.1), (0.1, 0.0)])
def test_simplex_rejects_steps_that_do_not_tile_or_allow_zero_weights(step: float, minimum: float) -> None:
    from models.fusion_objective import simplex_points

    with pytest.raises(ValueError):
        simplex_points(("a", "b"), step=step, minimum=minimum)


def test_grid_search_breaks_ties_toward_smallest_weights() -> None:
    best, value = grid_search({"a": (2.0, 1.0), "b": (3.0, 1.0)}, lambda p: 0.0)
    assert best == {"a": 1.0, "b": 1.0} and value == 0.0
    best, _ = grid_search({"a": (1.0, 2.0)}, lambda p: p["a"])
    assert best == {"a": 2.0}
    with pytest.raises(ValueError):
        grid_search({}, lambda p: 0.0)


def _cache() -> list[dict[str, Any]]:
    return [{"target": "SUN", "numeric_preserved": False, "raw_word": "5UN", "rows": [("SUN", -2.0, 0.5, [-1.0]), ("SON", -1.0, 1.0, [-6.0])]},
            {"target": "1945", "numeric_preserved": True, "raw_word": "1945", "rows": [("1945", -0.1, 0.0, [-9.0])]},
            {"target": "MOON", "numeric_preserved": False, "raw_word": "M00N", "rows": [("NOON", -1.0, 1.0, [-2.0])]}]


def test_bert_objective_counts_only_influenceable_items() -> None:
    params = {"alpha": 1.0, "lambda_vis": 1.0, "lambda_bert": 1.0, "gamma": 1.0}
    sun, son = -2.0 - 1.0 - 0.5, -1.0 - 6.0 - 1.0
    assert tune_bert.bert_objective(_cache(), params) == pytest.approx(sun - math.log(math.exp(sun) + math.exp(son)))
    assert tune_bert.cached_accuracy(_cache(), params) == pytest.approx(2 / 3)
    best, value = tune_bert.tune(_cache())
    assert set(best) == {"alpha", "lambda_vis", "lambda_bert", "gamma"} and value <= 0.0


# ---------------------------------------------------------------- benchmark helpers


def test_hardware_metadata_has_required_keys() -> None:
    hw = bench_report.hardware_metadata()
    assert {"cpu_model", "logical_cores", "ram_gb", "gpu", "os", "python", "torch", "transformers", "torch_threads", "timing_device"} <= set(hw)
    assert hw["timing_device"] == "cpu" and hw["ram_gb"] > 0


def _ladder(raw: float, ngram: float, bert: float, abl: float, app: float | None = None) -> dict[str, Any]:
    rows = (("raw_fp32", raw), ("classical", ngram), ("bert_fp32", bert), ("ablation_fp32cnn_int8bert", abl), ("quantized_app", bert if app is None else app))
    return {k: {"word_acc_pct": v} for k, v in rows}


def _results(checks: list[dict[str, Any]]) -> dict[str, str]:
    return {c["check"]: c["result"] for c in checks}


def test_monotonic_check_also_covers_the_app_row() -> None:
    vision = {"char_cnn_fp32": {"acc_36_pct": 91.0}, "char_cnn_int8": {"acc_36_pct": 91.0}}
    checks = _results(bench_report.pass_fail_checks(vision, _ladder(60, 80, 85, 85, app=79), {"fp32_mb": 420.0, "int8_mb": 200.0}))
    assert checks["Raw < N-gram < BERT FP32 (word acc)"] == "PASS"
    assert checks["Raw < N-gram < quantized app (word acc)"] == "FAIL"


@pytest.mark.parametrize(("drop", "ratio", "expected"), [(1.5, 2.0, "PASS"), (1.51, 1.99, "FAIL")])
def test_threshold_checks_are_inclusive_at_the_boundary(drop: float, ratio: float, expected: str) -> None:
    vision = {"char_cnn_fp32": {"acc_36_pct": 91.0}, "char_cnn_int8": {"acc_36_pct": 91.0 - drop}}
    checks = _results(bench_report.pass_fail_checks(vision, _ladder(60, 80, 85.0, 85.0 - drop), {"fp32_mb": 400.0, "int8_mb": 400.0 / ratio}))
    assert {checks[k] for k in checks if "drop" in k or "reduction" in k} == {expected}


def test_numeric_failures_name_the_real_cause() -> None:
    record = {"id": "t", "target": "2000", "raw_int8": "20O0", "outputs": {"quantized_app": "COO"}, "target_in_pool_int8": False, "numeric_preserved_int8": False}
    why = bench_report.explain_failure(record)
    assert "digit" in why and "lexicon" in why and "edit-distance" not in why


def _numeric_record(raw: str, in_pool: bool, guard: bool, conf: float) -> dict[str, Any]:
    return {"id": "t", "target": "1985", "raw_int8": raw, "outputs": {"quantized_app": raw}, "target_in_pool_int8": in_pool, "numeric_preserved_int8": guard, "mean_conf_int8": conf}


def test_all_digit_misread_names_the_lexicon_not_letters() -> None:
    why = bench_report.explain_failure(_numeric_record("1965", in_pool=False, guard=False, conf=0.5))
    assert why == "CNN misread the digits as '1965' and the digit string is not in the word lexicon"


@pytest.mark.parametrize(("conf", "confident"), [(0.40, False), (0.69, False), (0.70, True), (0.95, True)])
def test_guard_reason_says_confident_only_above_the_threshold(conf: float, confident: bool) -> None:
    why = bench_report.explain_failure(_numeric_record("1965", in_pool=True, guard=True, conf=conf))
    assert "kept the" in why and "'1965'" in why and ("confident" in why) is confident


def test_ram_falls_back_when_the_os_call_fails() -> None:
    class FailingKernel:
        @staticmethod
        def GlobalMemoryStatusEx(_ptr: Any) -> int:
            return 0

    assert bench_report._windows_total_ram_gb(FailingKernel()) is None


def test_pass_fail_checks_report_both_outcomes() -> None:
    vision = {"char_cnn_fp32": {"acc_36_pct": 91.0}, "char_cnn_int8": {"acc_36_pct": 90.0}}
    ok = bench_report.pass_fail_checks(vision, _ladder(60, 80, 85, 84), {"fp32_mb": 420.0, "int8_mb": 200.0})
    assert [c["result"] for c in ok] == ["PASS"] * 5
    bad = bench_report.pass_fail_checks({"char_cnn_fp32": {"acc_36_pct": 91.0}, "char_cnn_int8": {"acc_36_pct": 89.0}}, _ladder(60, 85, 85, 83), {"fp32_mb": 420.0, "int8_mb": 300.0})
    assert [c["result"] for c in bad] == ["FAIL"] * 5


def _record(rid: str, target: str, app: str, in_pool: bool, numeric: bool = False) -> dict[str, Any]:
    return {"id": rid, "target": target, "raw_int8": app, "outputs": {"quantized_app": app, "bert_fp32": app}, "target_in_pool_int8": in_pool, "numeric_preserved_int8": numeric}


def test_select_failures_prefers_reranker_errors_and_explains_them() -> None:
    records = [_record("a", "SUN", "SUN", True), _record("b", "MOON", "NOON", True), _record("c", "STAR", "5TAB", False), _record("d", "1985", "1965", True, numeric=True)]
    picked = bench_report.select_failures(records, minimum=3)
    assert [p["id"] for p in picked] == ["b", "d", "c"]
    assert "not among the 20 pool candidates" in picked[2]["why"] and "numeric guard" in picked[1]["why"] and "ranked 'NOON' higher" in picked[0]["why"]


def test_word_scores_and_row_summary() -> None:
    assert bench_report.word_scores("science", "SCIENCE") == (True, 7)
    assert bench_report.word_scores("SC1ENCE", "SCIENCE") == (False, 6)
    row = bench_report.summarize_row([True, False], [7, 6], 14, [10.0, 30.0], 5.0)
    assert row == {"word_acc_pct": 50.0, "char_acc_pct": pytest.approx(100 * 13 / 14), "size_mb": 5.0, "latency_mean_ms": 20.0, "n": 2}


def test_render_markdown_contains_both_tables_and_checks() -> None:
    vis = {"char_cnn_fp32": {"acc_47_pct": 89.0, "acc_36_pct": 91.0, "disk_mb": 6.5, "latency_b1_ms": 1.0, "latency_b8_ms": 2.0}}
    result = {"hardware": bench_report.hardware_metadata(), "vision": vis, "vision_n": 18800, "ladder_n": 300, "ladder": {"raw_fp32": {"label": "(1) raw", "word_acc_pct": 60.0, "char_acc_pct": 89.0, "size_mb": 6.5, "latency_mean_ms": 8.0}},
              "recall_pct": {"fp32": 90.0, "int8": 89.0}, "strip_char_acc_pct": 89.0, "strip_gap_note": "gap +2.00 pp (within 5.0 pp)", "bert_sizes": {"fp32_mb": 418.0, "int8_mb": 196.0},
              "bert_latency_ms": {"fp32": 60.0, "int8": 26.0}, "checks": [{"check": "x", "result": "PASS", "value": "1"}]}
    md = bench_report.render_markdown(result)
    assert "## Table 1" in md and "## Table 2" in md and "| (1) raw | 60.00 |" in md and "| x | PASS | 1 |" in md and "N=300" in md


@needs_emnist_cnn
def test_item_to_box_probs_returns_one_row_per_character() -> None:
    from data.prepare_data import load_emnist_dataset
    from data.word_pipeline import item_to_box_probs
    from models.char_models import load_char_cnn_fp32

    item = json.loads(DEV_PATH.read_text(encoding="utf-8"))[0]
    probs = item_to_box_probs(item, load_emnist_dataset(REPO_ROOT / "data" / "emnist", train=True), load_char_cnn_fp32(CNN_WEIGHTS))
    assert probs.shape == (len(item["target_word"]), 36) and np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    with pytest.raises(ValueError):
        item_to_box_probs({**item, "target_word": ""}, None, None)  # type: ignore[arg-type]


@needs_emnist_cnn
def test_run_item_runs_every_ladder_row_on_one_strip(tiny_reranker: BertMLMReranker, monkeypatch: pytest.MonkeyPatch) -> None:
    from data.prepare_data import load_emnist_dataset
    from data.word_pipeline import item_to_strip
    from models.char_models import load_char_cnn_fp32
    from models.classical_nlp import ClassicalNLPCorrector, tokenize
    from models.quantization import load_quantized_cnn

    item = json.loads(DEV_PATH.read_text(encoding="utf-8"))[1]
    sents = [tokenize(s) for s in ("THE SUN IS HOT", "THE MOON IS COLD", item["sentence"].replace("[MASK]", item["target_word"]))]
    lm = ClassicalNLPCorrector().train(sents, {w: 1 for s in sents for w in s if w.isalpha() and len(w) > 1})
    models = {"cnn_fp32": load_char_cnn_fp32(CNN_WEIGHTS), "cnn_int8": load_quantized_cnn(REPO_ROOT / "weights" / "char_cnn_int8.pt"), "ngram": lm,
              "bert_fp32": tiny_reranker, "bert_int8": tiny_reranker.to_int8(), "params": {"alpha": 1.0, "lambda_vis": 1.0, "lambda_bert": 1.0, "gamma": 1.0}}
    record = evaluate_all.run_item(item, item_to_strip(item, load_emnist_dataset(REPO_ROOT / "data" / "emnist", train=True)), models)
    assert set(record["outputs"]) == set(evaluate_all._LABELS) == set(record["latency_ms"])
    assert all(v > 0 for v in record["latency_ms"].values())
    assert record["latency_ms"]["bert_fp32"] > record["latency_ms"]["raw_fp32"]
    seen: dict[str, Any] = {}
    real_rank, real_rerank = models["ngram"].rank_pool, BertMLMReranker.rerank_candidates
    monkeypatch.setattr(models["ngram"], "rank_pool", lambda s, pool, meta: seen.setdefault("classical", pool) and real_rank(s, pool, meta))

    def spy(self: BertMLMReranker, s: str, pool: Any, **kw: Any) -> Any:
        seen.setdefault(f"bert_{len(seen)}", pool)
        return real_rerank(self, s, pool, **kw)

    monkeypatch.setattr(BertMLMReranker, "rerank_candidates", spy)
    evaluate_all.run_item(item, item_to_strip(item, load_emnist_dataset(REPO_ROOT / "data" / "emnist", train=True)), models)
    assert seen["classical"] is seen["bert_1"] is seen["bert_2"]  # rows 2, 3 and the ablation rank the identical pool


def test_long_context_is_truncated_and_the_length_guard_remains(tiny_reranker: BertMLMReranker, monkeypatch: pytest.MonkeyPatch) -> None:
    sentence = "the " * 70 + "[MASK]"
    assert set(tiny_reranker.span_log_probs(sentence, ["sun"])) == {"sun"}  # truncated to the window, not rejected
    monkeypatch.setattr(bert_reranker, "MAX_CONTEXT_PIECES_PER_SIDE", 10_000)  # without the window the guard still fires
    with pytest.raises(ValueError, match="WordPieces"):
        tiny_reranker.span_log_probs(sentence, ["sun"])


@pytest.mark.parametrize("module", [evaluate_all, tune_bert])
def test_cli_errors_print_one_line(module: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise IsADirectoryError(21, f"Expected a file, got a directory while reading: {tmp_path}", str(tmp_path))

    monkeypatch.setattr(module, "evaluate" if module is evaluate_all else "run", boom)
    assert module.main([]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and str(tmp_path) in err[0]


def test_batched_masked_positions_match_full_logits_on_seventy_sequences(tiny_reranker: BertMLMReranker) -> None:
    gen = torch.Generator().manual_seed(3)
    seqs = [[2, *torch.randint(5, len(VOCAB), (int(n),), generator=gen).tolist(), 3] for n in torch.randint(3, 20, (70,), generator=gen)]
    positions = [int(torch.randint(1, len(s) - 1, (1,), generator=gen)) for s in seqs]
    got = tiny_reranker._forward(seqs, positions)
    width = max(len(s) for s in seqs)
    ids = torch.zeros((70, width), dtype=torch.long)
    mask = torch.zeros((70, width), dtype=torch.long)
    for row, seq in enumerate(seqs):
        ids[row, : len(seq)] = torch.tensor(seq)
        mask[row, : len(seq)] = 1
    with torch.no_grad():
        full = torch.log_softmax(tiny_reranker.model(input_ids=ids, attention_mask=mask).logits, dim=-1)
    reference = full[torch.arange(70), torch.tensor(positions)]
    assert got.shape == (70, len(VOCAB)) and torch.allclose(got, reference, atol=1e-5)


def test_bert_is_loaded_at_the_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def capture(*_a: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        raise OSError("stop after recording the call")

    monkeypatch.setattr(bert_reranker.AutoTokenizer, "from_pretrained", capture)
    with pytest.raises(OSError):
        BertMLMReranker()
    assert calls and calls[0]["revision"] == bert_reranker.BERT_REVISION and len(bert_reranker.BERT_REVISION) == 40


def _recording(rr: BertMLMReranker, monkeypatch: pytest.MonkeyPatch) -> list[list[int]]:
    seen: list[list[int]] = []
    real = BertMLMReranker._forward

    def record(self: BertMLMReranker, sequences: Any, positions: Any) -> Any:
        seen.extend(list(s) for s in sequences)
        return real(self, sequences, positions)

    monkeypatch.setattr(BertMLMReranker, "_forward", record)
    return seen


def test_long_sentences_are_accepted_with_a_bounded_context_window(tiny_reranker: BertMLMReranker, monkeypatch: pytest.MonkeyPatch) -> None:
    sentence = ("the sun is hot " * 30)[:450] + " [MASK] " + "the moon is cold"
    assert len(sentence) <= 500
    seen = _recording(tiny_reranker, monkeypatch)
    out = tiny_reranker.span_log_probs(sentence, ["sun", "silence"])
    window = bert_reranker.MAX_CONTEXT_PIECES_PER_SIDE
    assert set(out) == {"sun", "silence"}
    assert all(len(seq) <= 2 * window + 3 + 2 for seq in seen)  # window each side + longest candidate (3 pieces) + [CLS]/[SEP]


def test_short_sentences_are_built_exactly_as_without_the_window(tiny_reranker: BertMLMReranker, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bert_reranker, "MAX_CONTEXT_PIECES_PER_SIDE", 10_000)
    unbounded = _recording(tiny_reranker, monkeypatch)
    expected = tiny_reranker.span_log_probs("the sun is hot and the [MASK] is cold", ["moon", "silence"])
    monkeypatch.undo()
    bounded = _recording(tiny_reranker, monkeypatch)
    got = tiny_reranker.span_log_probs("the sun is hot and the [MASK] is cold", ["moon", "silence"])
    assert bounded == unbounded and got == expected

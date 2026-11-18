"""BERT masked-LM re-ranker: batched span pseudo-log-likelihood fused with visual evidence.

Each candidate word w is WordPiece-tokenised into s_1..s_k. Single-piece candidates are all scored
from ONE forward pass over the sentence with a single [MASK]; multi-piece candidates are scored
from ONE batched forward pass over every (candidate, position j) sequence in which piece j is
masked and the other pieces are visible:

    PLL_norm(w) = (1 / k**alpha) * sum_j log P(s_j | context, s_-j)

and the fused score is lambda_vis * vis_log_prob + lambda_bert * PLL_norm - gamma * weighted_edit_distance.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from transformers import AutoTokenizer, BertForMaskedLM

from models.cloze import MASK_TOKEN, normalize_masked_sentence
from models.quantization import quantize_bert_dynamic

DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_FUSION_PARAMS: dict[str, float] = {"alpha": 1.0, "lambda_vis": 1.0, "lambda_bert": 1.0, "gamma": 1.0}
_PARAM_NAMES = ("alpha", "lambda_vis", "lambda_bert", "gamma")

__all__ = ["DEFAULT_FUSION_PARAMS", "DEFAULT_MODEL_NAME", "BertMLMReranker", "fused_score", "normalize_masked_sentence", "pll_norm"]


def pll_norm(piece_log_probs: Sequence[float], alpha: float) -> float:
    """Length-normalised span pseudo-log-likelihood: sum of piece log-probs divided by k**alpha."""
    if not piece_log_probs:
        raise ValueError("A candidate needs at least one WordPiece.")
    return float(sum(piece_log_probs)) / (len(piece_log_probs) ** float(alpha))


def fused_score(vis_log_prob: float, piece_log_probs: Sequence[float], weighted_edit_distance: float, params: Mapping[str, float]) -> float:
    """lambda_vis * vis + lambda_bert * PLL_norm - gamma * weighted edit distance."""
    bert = pll_norm(piece_log_probs, params["alpha"])
    return float(params["lambda_vis"] * vis_log_prob + params["lambda_bert"] * bert - params["gamma"] * weighted_edit_distance)


def _validate_params(params: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(params, Mapping) or any(name not in params for name in _PARAM_NAMES):
        raise ValueError(f"fusion params must provide {', '.join(_PARAM_NAMES)}.")
    return {name: float(params[name]) for name in _PARAM_NAMES}


class BertMLMReranker:
    """Scores candidate words for a cloze blank with a (FP32 or dynamic-INT8) BertForMaskedLM on CPU."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu", *, model: nn.Module | None = None, tokenizer: Any = None) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        loaded = model if model is not None else BertForMaskedLM.from_pretrained(model_name)
        self.model = loaded.to(self.device).eval()
        self.forward_calls = 0
        self._lowercase = bool(getattr(self.tokenizer, "do_lower_case", True))

    def to_int8(self) -> BertMLMReranker:
        """Return a NEW reranker wrapping a dynamic-INT8 copy of this model; this instance is unchanged."""
        int8_model = quantize_bert_dynamic(self.model)
        return BertMLMReranker(self.model_name, "cpu", model=int8_model, tokenizer=self.tokenizer)

    def word_pieces(self, word: str) -> list[int]:
        """WordPiece ids of a candidate word (lower-cased first for uncased vocabularies)."""
        text = word.lower() if self._lowercase else word
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"Candidate {word!r} produced no WordPieces.")
        return list(ids)

    def _context(self, sentence: str) -> tuple[list[int], list[int]]:
        left, right = normalize_masked_sentence(sentence).split(MASK_TOKEN)
        encode = self.tokenizer
        return list(encode(left, add_special_tokens=False)["input_ids"]), list(encode(right, add_special_tokens=False)["input_ids"])

    def _forward(self, sequences: Sequence[Sequence[int]]) -> torch.Tensor:
        """One (batched) forward pass; returns log-softmax over the vocabulary, shape (B, T, V)."""
        width = max(len(s) for s in sequences)
        limit = int(getattr(self.model.config, "max_position_embeddings", width))
        if width > limit:
            raise ValueError(f"Sentence plus candidate is {width} WordPieces; the model accepts at most {limit}.")
        self.forward_calls += 1
        pad = int(self.tokenizer.pad_token_id)
        ids = torch.full((len(sequences), width), pad, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, seq in enumerate(sequences):
            ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[row, : len(seq)] = 1
        with torch.no_grad():
            logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits
        return torch.log_softmax(logits.float(), dim=-1).cpu()

    def span_log_probs(self, sentence: str, words: Sequence[str]) -> dict[str, list[float]]:
        """Per-piece log P(s_j | ...) for each word: one pass for all 1-piece words, one batched pass for the rest."""
        prefix, suffix = self._context(sentence)
        cls_id, sep_id, mask_id = int(self.tokenizer.cls_token_id), int(self.tokenizer.sep_token_id), int(self.tokenizer.mask_token_id)
        pieces = {w: self.word_pieces(w) for w in dict.fromkeys(words)}
        out: dict[str, list[float]] = {}
        singles = [w for w, p in pieces.items() if len(p) == 1]
        if singles:
            log_probs = self._forward([[cls_id, *prefix, mask_id, *suffix, sep_id]])[0, len(prefix) + 1]
            out.update({w: [float(log_probs[pieces[w][0]])] for w in singles})
        jobs = [(w, j) for w, p in pieces.items() if len(p) > 1 for j in range(len(p))]
        if jobs:
            seqs = [[cls_id, *prefix, *pieces[w][:j], mask_id, *pieces[w][j + 1 :], *suffix, sep_id] for w, j in jobs]
            log_probs = self._forward(seqs)
            for row, (w, j) in enumerate(jobs):
                out.setdefault(w, []).append(float(log_probs[row, len(prefix) + 1 + j, pieces[w][j]]))
        return out

    def rerank_candidates(self, sentence: str, candidates: Sequence[Mapping[str, Any]], *, params: Mapping[str, float] = DEFAULT_FUSION_PARAMS,
                          raw_word: str | None = None, numeric_preserved: bool = False) -> dict[str, Any]:
        """Fuse visual, BERT and edit evidence over a candidate pool (from ClassicalNLPCorrector.generate_candidates)."""
        if not candidates:
            raise ValueError("candidates must be a non-empty sequence.")
        weights = _validate_params(params)
        span = self.span_log_probs(sentence, [str(c["word"]) for c in candidates])
        rows = [self._score_row(c, span[str(c["word"])], weights) for c in candidates]
        rows.sort(key=lambda r: r["word"])
        rows.sort(key=lambda r: (numeric_preserved and r["word"] == raw_word, r["score"]), reverse=True)
        return {"predicted_word": rows[0]["word"], "raw_word": raw_word, "numeric_preserved": numeric_preserved, "top_candidates": rows}

    @staticmethod
    def _score_row(cand: Mapping[str, Any], piece_lps: list[float], weights: Mapping[str, float]) -> dict[str, Any]:
        norm = pll_norm(piece_lps, weights["alpha"])
        score = fused_score(float(cand["vis_log_prob"]), piece_lps, float(cand["weighted_edit_distance"]), weights)
        return {**cand, "bert_piece_log_probs": piece_lps, "bert_pieces": len(piece_lps), "bert_pll_norm": norm, "score": score}

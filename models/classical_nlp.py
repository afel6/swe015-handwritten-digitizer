"""Classical NLP: tokenization, SymSpell, interpolated (Jelinek-Mercer) n-gram LM, and lattice fusion."""

from __future__ import annotations

import errno
import io
import math
import os
import pickle
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.preprocess import CANONICAL_36_CHARS

CONFUSION_MAP: dict[str, tuple[str, ...]] = {ch: tuple(o for o in g if o != ch) for g in ("0O", "1IL", "2Z", "4A", "5S", "6G", "8B", "9Q") for ch in g}
DEFAULT_SMOOTHING_K, DEFAULT_WEIGHTS_PATH = 0.05, Path("weights/ngram_lm.pkl")
START_TOKEN, END_TOKEN, UNK_TOKEN, MASK_TOKEN = "<s>", "</s>", "<UNK>", "[MASK]"
_START_ID, _NUM_CHARS, _BYTES_PER_MB = -1, len(CANONICAL_36_CHARS), 1024.0 * 1024.0
_CHAR_TO_IDX = {ch: idx for idx, ch in enumerate(CANONICAL_36_CHARS)}
_TOKEN_RE, _UNDERSCORE_MASK_RE, _CANONICAL_LEX_RE = re.compile(r"[A-Z0-9]+"), re.compile(r"_{3,}"), re.compile(r"^(?:[A-Z]{2,}|A|I)$")
_ANY_CASE_MASK_RE, MAX_SENTENCE_CHARS = re.compile(re.escape(MASK_TOKEN), re.IGNORECASE), 500
_MAX_SYMSPELL_ED, _SYMSPELL_PREFIX_LEN, _MAX_LATTICE_BOXES, _MAX_BEAM_SEEDS = 2, 10, 6, 64
_DEFAULT_MAX_CANDIDATES, _CANDIDATE_POOL_MIN, _GRID_STEPS = 20, 60, 20
_PROB_EPS, _CONFUSION_PENALTY, _INDEL_VIS_PENALTY = 1e-12, 1.5, 4.0
_NUMERIC_CONF_THRESHOLD, _NUMERIC_MAX_LEX_WED = 0.70, 1.5
_SUB_CONFUSION_COST, _SUB_OTHER_COST, _INDEL_COST, _MIN_L0_SMOOTH, _LAMBDA_SUM_TOL = 0.5, 1.0, 1.5, 0.005, 1e-5
DEFAULT_LAMBDAS: tuple[float, float, float, float] = (0.55, 0.30, 0.14, 0.01)
DEFAULT_BIGRAM_LAMBDAS: tuple[float, float, float, float] = (0.0, 0.75, 0.24, 0.01)
DEFAULT_UNIGRAM_LAMBDAS: tuple[float, float, float, float] = (0.0, 0.0, 0.99, 0.01)
DEFAULT_FUSION_WEIGHTS: dict[str, float] = {"vis_weight": 1.0, "edit_penalty": 2.0, "lm_weight": 1.0}
_STATE_FIELDS = ("vocab", "lexicon", "lexicon_words", "deletes", "unigram_counts", "total_tokens", "bigram_counts", "bigram_totals", "trigram_counts", "trigram_totals", "smoothing_k", "lambdas", "bigram_lambdas", "unigram_lambdas", "fusion_weights")


def tokenize(text: str) -> list[str]:
    """Tokenize text into uppercase alphanumeric tokens."""
    if not isinstance(text, str): raise TypeError("text must be a string.")
    return _TOKEN_RE.findall(text.upper())


def normalize_word(word: str) -> str:
    """Normalize a word to uppercase alphanumeric characters only."""
    if not isinstance(word, str): raise TypeError("word must be a string.")
    return "".join(_TOKEN_RE.findall(word.upper()))


def extract_cloze_context(sentence: str) -> tuple[list[str], list[str]]:
    """Normalize '___' and any-case '[mask]' blanks to [MASK], require exactly one, and tokenize both sides."""
    if not isinstance(sentence, str): raise TypeError("sentence must be a string.")
    if len(sentence) > MAX_SENTENCE_CHARS: raise ValueError(f"sentence must be at most {MAX_SENTENCE_CHARS} characters.")
    parts = _ANY_CASE_MASK_RE.sub(MASK_TOKEN, _UNDERSCORE_MASK_RE.sub(MASK_TOKEN, sentence)).split(MASK_TOKEN)
    if len(parts) != 2: raise ValueError("sentence must contain exactly one [MASK] (or '___') blank.")
    return tokenize(parts[0]), tokenize(parts[1])


def _levenshtein(a: str, b: str, indel: float, weighted: bool) -> float:
    if not isinstance(a, str) or not isinstance(b, str): raise TypeError("Edit distance inputs must be strings.")
    if len(a) < len(b): a, b = b, a
    prev = [j * indel for j in range(len(b) + 1)]
    for i, ca in enumerate(a):
        curr = [(i + 1) * indel]
        for j, cb in enumerate(b):
            sub = 0.0 if ca == cb else (_SUB_CONFUSION_COST if weighted and cb in CONFUSION_MAP.get(ca, ()) else _SUB_OTHER_COST)
            curr.append(min(curr[-1] + indel, prev[j + 1] + indel, prev[j] + sub))
        prev = curr
    return float(prev[-1])


def unweighted_edit_distance(a: str, b: str) -> int:
    """Compute standard unweighted Levenshtein edit distance."""
    return int(_levenshtein(a, b, 1.0, False))


def weighted_edit_distance(a: str, b: str) -> float:
    """Compute OCR-confusion-weighted Levenshtein distance."""
    return _levenshtein(a, b, _INDEL_COST, True)


def _generate_deletes(text: str) -> set[str]:
    deletes, frontier = {text}, {text}
    for _ in range(_MAX_SYMSPELL_ED):
        if not (nxt := {s[:i] + s[i + 1 :] for s in frontier for i in range(len(s))} - deletes): break
        deletes.update(nxt); frontier = nxt
    return deletes


def _validate_probs_36(probs_36: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(probs_36, torch.Tensor):
        if probs_36.is_complex() or probs_36.dtype == torch.bool: raise ValueError("probs_36 must contain real numeric values.")
        arr = probs_36.detach().cpu().numpy()
    elif probs_36 is None or isinstance(probs_36, (str, bytes, bytearray, dict, bool)):
        raise TypeError("probs_36 must be a 2D array or tensor of shape (L, 36).")
    else:
        arr = np.asarray(probs_36)
    if arr.ndim != 2 or arr.shape[1] != _NUM_CHARS or arr.shape[0] < 1 or not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating) or arr.dtype == np.bool_:
        raise ValueError("probs_36 must be a real numeric array of shape (L, 36) with L >= 1.")
    work = np.array(arr, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(work)) or np.any(work < 0): raise ValueError("probs_36 must contain finite non-negative values.")
    sums = work.sum(axis=1, keepdims=True)
    return np.where(sums <= 0, 1.0 / _NUM_CHARS, work / np.where(sums <= 0, 1.0, sums))


def _validate_lambdas(lambdas: Sequence[float]) -> tuple[float, float, float, float]:
    if isinstance(lambdas, (str, bytes)) or not isinstance(lambdas, Sequence) or len(lambdas) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in lambdas):
        raise ValueError("lambdas must be a 4-tuple of real numbers.")
    vals = tuple(float(x) for x in lambdas)
    if any(not math.isfinite(x) or x < 0.0 for x in vals) or abs(sum(vals) - 1.0) > _LAMBDA_SUM_TOL:
        raise ValueError("lambdas must be finite, non-negative, and sum to 1.0.")
    return vals  # type: ignore[return-value]


def _validate_sentences(sentences: Any, name: str) -> list[list[str]]:
    if isinstance(sentences, (str, bytes)) or not isinstance(sentences, Sequence) or any(isinstance(s, (str, bytes)) or not isinstance(s, Sequence) or any(not isinstance(t, str) for t in s) for s in sentences):
        raise TypeError(f"{name} must be a sequence of string-token sentences.")
    if not (out := [list(s) for s in sentences if s]): raise ValueError(f"{name} must contain at least one non-empty sentence.")
    return out


def _map_os_error(err: OSError, target: Path, action: str) -> OSError:
    label = os.fspath(target)
    if err.errno == errno.ENOENT: return FileNotFoundError(errno.ENOENT, f"File not found while {action}: {label}", label)
    if err.errno == errno.EISDIR or (err.errno == errno.EACCES and target.is_dir()): return IsADirectoryError(errno.EISDIR, f"Expected a file path, got a directory: {label}", label)
    if err.errno == errno.EACCES: return PermissionError(errno.EACCES, f"Permission denied while {action}: {label}", label)
    return OSError(err.errno, f"Failed {action} {label}: {err.strerror or 'OS error'}", label)


def _validate_path(path: str | os.PathLike[str]) -> Path:
    if not isinstance(path, (str, os.PathLike)): raise TypeError("Model path must be a string or path-like object.")
    raw = os.fspath(path)
    if not isinstance(raw, str): raise TypeError("Model path must resolve to a string.")
    if not raw: raise ValueError("Model path must not be empty.")
    return Path(raw)


def _pickle_io(path: str | os.PathLike[str], mode: str, payload: dict[str, Any] | None = None) -> tuple[Path, Any]:
    target, action = _validate_path(path), "saving model" if "w" in mode else "loading model"
    try:
        with open(target, mode) as handle:
            if "w" in mode: pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL); return target, None
            return target, pickle.load(handle)
    except OSError as err:
        raise _map_os_error(err, target, action) from err
    except (pickle.UnpicklingError, EOFError, AttributeError, ValueError, TypeError) as err:
        raise ValueError(f"Invalid classical NLP model file: {target}") from err


class ClassicalNLPCorrector:
    """SymSpell + linear-interpolated trigram LM + visual/confusion candidate fusion."""

    def __init__(self) -> None:
        self.vocab: list[str] = []
        self.word_to_id, self.lexicon, self.lexicon_words, self.deletes = {}, {}, [], {}
        self.unigram_counts, self._p_uni = np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)
        self.bigram_counts, self.bigram_totals, self.trigram_counts, self.trigram_totals = {}, {}, {}, {}
        self.total_tokens, self.smoothing_k, self._unk_id, self._end_id, self._trained, self._model_path = 0, DEFAULT_SMOOTHING_K, 0, 1, False, None
        self.lambdas, self.bigram_lambdas, self.unigram_lambdas = DEFAULT_LAMBDAS, DEFAULT_BIGRAM_LAMBDAS, DEFAULT_UNIGRAM_LAMBDAS
        self.fusion_weights = dict(DEFAULT_FUSION_WEIGHTS)

    def _require_trained(self) -> None:
        if not self._trained: raise RuntimeError("ClassicalNLPCorrector must be trained or loaded before use.")

    def _parse_history(self, history: Any) -> tuple[int, int]:
        if not isinstance(history, (tuple, list)) or len(history) != 2 or not all(isinstance(h, str) for h in history):
            raise ValueError("history must be a 2-tuple of strings.")
        return tuple(_START_ID if h == START_TOKEN else self.word_to_id.get(h, self._unk_id) for h in history)  # type: ignore[return-value]

    def _order_probs(self, uid: int, vid: int, wid: int) -> tuple[float, float, float]:
        """Jelinek-Mercer components: add-k unigram, MLE bigram/trigram backing off to the lower order for unseen contexts."""
        tri_tot, bi_tot = self.trigram_totals.get((uid, vid), 0), self.bigram_totals.get(vid, 0)
        p1 = float(self._p_uni[wid])
        p2 = self.bigram_counts[vid].get(wid, 0) / bi_tot if bi_tot else p1
        return (self.trigram_counts[(uid, vid)].get(wid, 0) / tri_tot if tri_tot else p2), p2, p1

    def _build_symspell_index(self, lexicon: Mapping[str, int]) -> None:
        if not isinstance(lexicon, Mapping): raise TypeError("lexicon must be a mapping of word -> count.")
        clean = {}
        for word, cnt in lexicon.items():
            if not isinstance(word, str) or isinstance(cnt, bool) or not isinstance(cnt, int) or cnt < 0:
                raise ValueError("lexicon must map strings to non-negative integers.")
            if _CANONICAL_LEX_RE.fullmatch(word): clean[word] = cnt
        if not clean: raise ValueError("lexicon must contain at least one canonical word.")
        self.lexicon, self.lexicon_words, self.deletes = clean, sorted(clean), {}
        for wid, word in enumerate(self.lexicon_words):
            for key in _generate_deletes(word[:_SYMSPELL_PREFIX_LEN]): self.deletes.setdefault(key, []).append(wid)

    def _symspell_lookup(self, query: str) -> list[str]:
        found: set[int] = set()
        for key in _generate_deletes(query[:_SYMSPELL_PREFIX_LEN]):
            for wid in self.deletes.get(key, ()):
                cand = self.lexicon_words[wid]
                if wid not in found and abs(len(cand) - len(query)) <= _MAX_SYMSPELL_ED and unweighted_edit_distance(query, cand) <= _MAX_SYMSPELL_ED:
                    found.add(wid)
        return [self.lexicon_words[i] for i in found]

    def train(self, train_sentences: Sequence[Sequence[str]], lexicon: Mapping[str, int], k: float = DEFAULT_SMOOTHING_K, max_vocab_size: int = 28000) -> ClassicalNLPCorrector:
        """Train SymSpell index, an add-k smoothed unigram, and bigram/trigram counts for interpolation."""
        if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(float(k)) or float(k) <= 0.0:
            raise ValueError("k must be a finite positive float.")
        if isinstance(max_vocab_size, bool) or not isinstance(max_vocab_size, int) or max_vocab_size < 2:
            raise ValueError("max_vocab_size must be an integer >= 2.")
        non_empty = _validate_sentences(train_sentences, "train_sentences")
        self._build_symspell_index(lexicon)
        freq = Counter(tok for sent in non_empty for tok in sent)
        ordered = sorted((set(freq) | set(self.lexicon)) - {START_TOKEN, END_TOKEN, UNK_TOKEN}, key=lambda w: (0 if w in self.lexicon else 1, -freq.get(w, 0), w))
        self.vocab = [UNK_TOKEN, END_TOKEN, *ordered[: max_vocab_size - 2]]
        self.word_to_id = {w: i for i, w in enumerate(self.vocab)}
        self._unk_id, self._end_id, v_size, k_val = 0, 1, len(self.vocab), float(k)
        uni, bi, bi_tot, tri, tri_tot = np.zeros((v_size,), dtype=np.int64), {}, {}, {}, {}
        for sent in non_empty:
            ids = [self.word_to_id.get(tok, self._unk_id) for tok in sent] + [self._end_id]
            for idx, wid in enumerate(ids):
                uni[wid] += 1; u, v = (_START_ID if idx < 2 else ids[idx - 2]), (_START_ID if idx < 1 else ids[idx - 1])
                br = bi.setdefault(v, {}); br[wid] = br.get(wid, 0) + 1; bi_tot[v] = bi_tot.get(v, 0) + 1
                tr = tri.setdefault((u, v), {}); tr[wid] = tr.get(wid, 0) + 1; tri_tot[(u, v)] = tri_tot.get((u, v), 0) + 1
        self.unigram_counts, self.total_tokens, self.smoothing_k, self._trained = uni, int(uni.sum()), k_val, True
        self.bigram_counts, self.bigram_totals, self.trigram_counts, self.trigram_totals = bi, bi_tot, tri, tri_tot
        self._p_uni = (uni.astype(np.float64) + k_val) / (self.total_tokens + k_val * v_size)
        return self

    @staticmethod
    def _mle_dist(counts: Mapping[int, int] | None, total: int, fallback: np.ndarray) -> np.ndarray:
        if not total: return fallback.copy()
        dist = np.zeros_like(fallback)
        for wid, cnt in (counts or {}).items(): dist[wid] = cnt / total
        return dist

    def next_word_distribution(self, history: tuple[str, str], lambdas: tuple[float, float, float, float] | None = None) -> np.ndarray:
        """Return normalized float64 distribution over V for history (u, v)."""
        self._require_trained(); uid, vid = self._parse_history(history)
        l3, l2, l1, l0 = self.lambdas if lambdas is None else _validate_lambdas(lambdas)
        d2 = self._mle_dist(self.bigram_counts.get(vid), self.bigram_totals.get(vid, 0), self._p_uni)
        d3 = self._mle_dist(self.trigram_counts.get((uid, vid)), self.trigram_totals.get((uid, vid), 0), d2)
        return l3 * d3 + l2 * d2 + l1 * self._p_uni + (l0 / len(self.vocab))

    def cond_prob(self, word: str, history: tuple[str, str], lambdas: tuple[float, float, float, float] | None = None) -> float:
        """Compute O(1) interpolated P(word | u, v) matching next_word_distribution."""
        self._require_trained()
        if not isinstance(word, str): raise TypeError("word must be a string.")
        uid, vid = self._parse_history(history)
        l3, l2, l1, l0 = self.lambdas if lambdas is None else _validate_lambdas(lambdas)
        p3, p2, p1 = self._order_probs(uid, vid, self.word_to_id.get(word, self._unk_id))
        return float(l3 * p3 + l2 * p2 + l1 * p1 + l0 / len(self.vocab))

    def _extract_eval_probs(self, sentences: Sequence[Sequence[str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        p3_l, p2_l, p1_l = [], [], []
        for sent in _validate_sentences(sentences, "sentences"):
            ids = [self.word_to_id.get(t, self._unk_id) for t in sent] + [self._end_id]
            for idx, wid in enumerate(ids):
                p3, p2, p1 = self._order_probs(_START_ID if idx < 2 else ids[idx - 2], _START_ID if idx < 1 else ids[idx - 1], wid)
                p3_l.append(p3); p2_l.append(p2); p1_l.append(p1)
        return np.asarray(p3_l), np.asarray(p2_l), np.asarray(p1_l), 1.0 / len(self.vocab)

    def tune_lambdas(self, dev_sentences: Sequence[Sequence[str]]) -> tuple[float, float, float, float]:
        """Grid-search trigram and bigram interpolation weights on dev_sentences."""
        self._require_trained(); p3, p2, p1, p0 = self._extract_eval_probs(dev_sentences)
        best_tri, best_tri_nll, best_bi, best_bi_nll = self.lambdas, math.inf, self.bigram_lambdas, math.inf
        for l0 in (_MIN_L0_SMOOTH, 0.01, 0.02, 0.05):
            rem = 1.0 - l0
            for i in range(1, _GRID_STEPS):
                b2 = round(rem * (i / _GRID_STEPS), 6)
                if (b1 := round(rem - b2, 6)) > 0.0 and (nll := -float(np.log(b2 * p2 + b1 * p1 + l0 * p0).mean())) < best_bi_nll:
                    best_bi_nll, best_bi = nll, (0.0, b2, b1, l0)
                for j in range(1, _GRID_STEPS - i):
                    l2 = round(rem * (j / _GRID_STEPS), 6)
                    if (l1 := round(1.0 - l0 - b2 - l2, 6)) > 0.0 and (nll := -float(np.log(b2 * p3 + l2 * p2 + l1 * p1 + l0 * p0).mean())) < best_tri_nll:
                        best_tri_nll, best_tri = nll, (b2, l2, l1, l0)
        self.lambdas, self.bigram_lambdas = best_tri, best_bi
        return self.lambdas

    def perplexity(self, sentences: Sequence[Sequence[str]], order: int = 3) -> float:
        """Compute exact closed-vocabulary perplexity for order 1, 2, or 3."""
        self._require_trained()
        if isinstance(order, bool) or order not in (1, 2, 3): raise ValueError("order must be 1, 2, or 3.")
        p3, p2, p1, p0 = self._extract_eval_probs(sentences)
        l3, l2, l1, l0 = {1: self.unigram_lambdas, 2: self.bigram_lambdas, 3: self.lambdas}[order]
        return float(math.exp(-float(np.log(l3 * p3 + l2 * p2 + l1 * p1 + l0 * p0).mean())))

    def score_cloze_context(self, word: str, left_tokens: Sequence[str], right_tokens: Sequence[str]) -> float:
        """Compute bidirectional log-probability of word inside padded left/right context."""
        self._require_trained()
        if not isinstance(word, str) or any(isinstance(s, (str, bytes)) or not isinstance(s, Sequence) for s in (left_tokens, right_tokens)):
            raise TypeError("word must be str and left_tokens/right_tokens must be sequences.")
        left, right = [START_TOKEN, START_TOKEN, *left_tokens], [*right_tokens, END_TOKEN, END_TOKEN]
        wm2, wm1, wp1, wp2 = left[-2], left[-1], right[0], right[1]
        return float(math.log(self.cond_prob(word, (wm2, wm1))) + math.log(self.cond_prob(wp1, (wm1, word))) + math.log(self.cond_prob(wp2, (word, wp1))))

    def _beam_search_seeds(self, probs_norm: np.ndarray) -> tuple[list[tuple[str, float]], str, float]:
        length = probs_norm.shape[0]
        top2 = np.argsort(-probs_norm, axis=1, kind="mergesort")[:, :2]
        p1, p2 = probs_norm[np.arange(length), top2[:, 0]], probs_norm[np.arange(length), top2[:, 1]]
        raw_word = "".join(CANONICAL_36_CHARS[int(idx)] for idx in top2[:, 0])
        margins = np.log(p1 + _PROB_EPS) - np.log(p2 + _PROB_EPS)
        branch = {idx for _, idx in sorted((float(margins[i]), i) for i in range(length))[:_MAX_LATTICE_BOXES]}
        beam: list[tuple[str, float]] = [("", 0.0)]
        for i in range(length):
            c1, s1 = raw_word[i], math.log(float(p1[i]) + _PROB_EPS); opts = {c1: s1}
            if i in branch:
                opts[CANONICAL_36_CHARS[int(top2[i, 1])]] = math.log(float(p2[i]) + _PROB_EPS)
                for alt in CONFUSION_MAP.get(c1, ()):
                    opts[alt] = max(opts.get(alt, -math.inf), max(math.log(float(probs_norm[i, _CHAR_TO_IDX[alt]]) + _PROB_EPS), s1 - _CONFUSION_PENALTY))
            nxt: dict[str, float] = {}
            for pref, sc in beam:
                for ch, ch_s in opts.items():
                    if (cand := pref + ch) not in nxt or sc + ch_s > nxt[cand]: nxt[cand] = sc + ch_s
            beam = sorted(nxt.items(), key=lambda kv: (-kv[1], kv[0]))[:_MAX_BEAM_SEEDS]
        return beam, raw_word, float(np.mean(p1))

    def _rank_candidates(self, items: list[dict[str, Any]], raw_word: str, num_pres: bool, limit: int) -> list[dict[str, Any]]:
        ordered = sorted(items, key=lambda c: c["word"])
        ordered.sort(key=lambda c: (1 if num_pres and c["word"] == raw_word else 0, c["score"], -c["weighted_edit_distance"], c["vis_log_prob"]), reverse=True)
        return ordered[:limit]

    def generate_candidates(self, probs_36: np.ndarray | torch.Tensor, max_candidates: int = _DEFAULT_MAX_CANDIDATES) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build bounded lattice seeds, SymSpell candidates, and numeric-guard metadata."""
        self._require_trained()
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 1: raise ValueError("max_candidates must be an integer >= 1.")
        probs_norm = _validate_probs_36(probs_36)
        length, log_p = probs_norm.shape[0], np.log(np.maximum(probs_norm, _PROB_EPS))
        seeds, raw_word, mean_conf = self._beam_search_seeds(probs_norm)
        best_vis: dict[str, float] = {}
        for seed, seed_lp in seeds:
            for cand in self._symspell_lookup(seed):
                vis_lp = float(sum(log_p[i, _CHAR_TO_IDX[ch]] for i, ch in enumerate(cand))) if len(cand) == length else float(seed_lp - abs(len(cand) - length) * _INDEL_VIS_PENALTY)
                if cand not in best_vis or vis_lp > best_vis[cand]: best_vis[cand] = vis_lp
        vw, ep = self.fusion_weights["vis_weight"], self.fusion_weights["edit_penalty"]
        items = [{"word": w, "score": float(vw * v - ep * weighted_edit_distance(raw_word, w)), "vis_log_prob": float(v), "edit_distance": unweighted_edit_distance(raw_word, w), "weighted_edit_distance": weighted_edit_distance(raw_word, w), "in_lexicon": w in self.lexicon, "is_numeric": w.isdigit()} for w, v in best_vis.items()]
        numeric_preserved = raw_word.isdigit() and ((mean_conf >= _NUMERIC_CONF_THRESHOLD) or (not any(c["in_lexicon"] and c["weighted_edit_distance"] <= _NUMERIC_MAX_LEX_WED for c in items)))
        if (raw_word.isdigit() and raw_word not in best_vis) or not items:
            raw_lp = float(sum(log_p[i, _CHAR_TO_IDX[ch]] for i, ch in enumerate(raw_word)))
            items.append({"word": raw_word, "score": float(vw * raw_lp), "vis_log_prob": raw_lp, "edit_distance": 0, "weighted_edit_distance": 0.0, "in_lexicon": raw_word in self.lexicon, "is_numeric": raw_word.isdigit()})
        return self._rank_candidates(items, raw_word, numeric_preserved, max_candidates), {"raw_word": raw_word, "mean_top1_conf": mean_conf, "numeric_preserved": numeric_preserved}

    def tune_fusion_weights(self, dev_items: Sequence[Mapping[str, Any]], probs_list: Sequence[np.ndarray | torch.Tensor]) -> dict[str, float]:
        """Grid-search visual/edit/LM fusion weights maximizing Dev word accuracy."""
        self._require_trained()
        if isinstance(dev_items, (str, bytes)) or isinstance(probs_list, (str, bytes)) or not dev_items or len(dev_items) != len(probs_list):
            raise ValueError("dev_items and probs_list must be non-empty sequences of matching length.")
        cached = []
        for item, probs in zip(dev_items, probs_list):
            if not isinstance(item, Mapping) or "sentence" not in item or "target_word" not in item: raise ValueError("Each dev_item must be a mapping with 'sentence' and 'target_word'.")
            left, right = extract_cloze_context(item["sentence"])
            cands, meta = self.generate_candidates(probs, max_candidates=_CANDIDATE_POOL_MIN)
            rows = [(c["word"], c["vis_log_prob"], c["weighted_edit_distance"], self.score_cloze_context(c["word"], left, right)) for c in cands]
            cached.append((normalize_word(item["target_word"]), bool(meta["numeric_preserved"]), meta["raw_word"], rows))
        best_acc, best_dist, best_w = -1, math.inf, dict(DEFAULT_FUSION_WEIGHTS)
        for vw in (0.5, 1.0, 1.5, 2.0):
            for ep in (0.5, 1.0, 2.0, 3.0, 4.0):
                for lw in (0.25, 0.5, 1.0, 1.5, 2.0):
                    hits = sum(1 for tgt, np_f, rw, rows in cached if (rw if np_f else max(rows, key=lambda r: (vw * r[1] - ep * r[2] + lw * r[3], -r[2], r[1]))[0]) == tgt)
                    dist = abs(vw - 1.0) + abs(ep - 2.0) + abs(lw - 1.0)
                    if hits > best_acc or (hits == best_acc and dist < best_dist): best_acc, best_dist, best_w = hits, dist, {"vis_weight": vw, "edit_penalty": ep, "lm_weight": lw}
        self.fusion_weights = best_w
        return dict(self.fusion_weights)

    def predict(self, sentence: str, probs_36: np.ndarray | torch.Tensor, max_candidates: int = _DEFAULT_MAX_CANDIDATES) -> dict[str, Any]:
        """Predict masked word using candidate lattice, SymSpell, N-gram LM, and numeric guard."""
        self._require_trained()
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 1: raise ValueError("max_candidates must be an integer >= 1.")
        left, right = extract_cloze_context(sentence)
        cands, meta = self.generate_candidates(probs_36, max_candidates=max(max_candidates, _CANDIDATE_POOL_MIN))
        vw, ep, lw = self.fusion_weights["vis_weight"], self.fusion_weights["edit_penalty"], self.fusion_weights["lm_weight"]
        scored = [{**c, "lm_log_prob": (lp := self.score_cloze_context(c["word"], left, right)), "score": float(vw * c["vis_log_prob"] - ep * c["weighted_edit_distance"] + lw * lp)} for c in cands]
        raw_word, num_pres = meta["raw_word"], bool(meta["numeric_preserved"])
        top = self._rank_candidates(scored, raw_word, num_pres, max_candidates)
        return {"predicted_word": top[0]["word"], "raw_word": raw_word, "numeric_preserved": num_pres, "top_candidates": top}

    def save(self, path: str | os.PathLike[str] = DEFAULT_WEIGHTS_PATH) -> Path:
        """Persist trained model state to disk with explicit OS error mapping."""
        self._require_trained()
        target, _ = _pickle_io(path, "wb", {k: getattr(self, k) for k in _STATE_FIELDS})
        self._model_path = target
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str] = DEFAULT_WEIGHTS_PATH) -> ClassicalNLPCorrector:
        """Load trained model state from disk with explicit OS error mapping."""
        target, payload = _pickle_io(path, "rb")
        if not isinstance(payload, dict) or not set(_STATE_FIELDS).issubset(payload): raise ValueError(f"Corrupted classical NLP model payload: {target}")
        obj = cls()
        for k in _STATE_FIELDS: setattr(obj, k, payload[k])
        obj.word_to_id = {w: i for i, w in enumerate(obj.vocab)}
        obj._unk_id, obj._end_id, obj._trained, obj._model_path = obj.word_to_id[UNK_TOKEN], obj.word_to_id[END_TOKEN], True, target
        obj._p_uni = (obj.unigram_counts.astype(np.float64) + obj.smoothing_k) / (obj.total_tokens + obj.smoothing_k * len(obj.vocab))
        return obj

    def get_model_size_mb(self, path: str | os.PathLike[str] | None = None) -> float:
        """Return serialized model size in megabytes."""
        target = _validate_path(path) if path is not None else self._model_path
        if target is not None and (path is not None or target.exists()):
            if target.is_dir(): raise IsADirectoryError(errno.EISDIR, f"Expected a file path, got a directory: {target}", os.fspath(target))
            try: return float(target.stat().st_size) / _BYTES_PER_MB
            except OSError as err: raise _map_os_error(err, target, "checking model size") from err
        self._require_trained()
        buf = io.BytesIO(); pickle.dump({k: getattr(self, k) for k in _STATE_FIELDS}, buf, protocol=pickle.HIGHEST_PROTOCOL)
        return float(buf.getbuffer().nbytes) / _BYTES_PER_MB

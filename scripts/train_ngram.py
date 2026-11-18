"""Train, tune, evaluate, and save the classical N-gram language model corrector."""

from __future__ import annotations

import argparse
import errno
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

if str(_ROOT := Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.prepare_data import build_lexicon, prepare_all, prepare_corpus
from data.preprocess import CANONICAL_36_CHARS
from models.classical_nlp import (
    CONFUSION_MAP,
    DEFAULT_SMOOTHING_K,
    DEFAULT_WEIGHTS_PATH,
    ClassicalNLPCorrector,
    normalize_word,
    tokenize,
)

_SEED, _CER, _BIAS, _N_CHARS = 42, 0.12, 0.85, len(CANONICAL_36_CHARS)
_METRICS_PATH = Path("results/ngram_metrics.json")
_DEV_PATH, _TEST_PATH = Path("data/worksheets_dev_50.json"), Path("data/worksheets_test_300.json")
_SPLIT_RE, _C2I = re.compile(r"[.!?\n]+"), {ch: i for i, ch in enumerate(CANONICAL_36_CHARS)}
_OTHERS = {ch: tuple(c for c in CANONICAL_36_CHARS if c != ch) for ch in CANONICAL_36_CHARS}
_NOISE = {"bg": (2e-4, 1e-3), "clean": (0.75, 0.94), "leak": (0.02, 0.08), "err": (0.5, 0.72), "true": (0.18, 0.35)}
_OS_ERRS = {errno.ENOENT: "Path not found", errno.EISDIR: "Target path is a directory",
            errno.ENOTDIR: "Parent path is not a directory", errno.EEXIST: "Path blocks directory creation",
            errno.EACCES: "Permission denied", errno.EPERM: "Operation not permitted", errno.ENOSPC: "Disk full"}


def _texts_to_sentences(texts: Sequence[str]) -> list[list[str]]:
    """Split documents on sentence boundaries and tokenize each non-empty sentence."""
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("Each corpus document must be a string.")
    return [toks for text in texts for chunk in _SPLIT_RE.split(text) if (toks := tokenize(chunk))]


def _load_worksheet_items(path: Path) -> list[dict[str, Any]]:
    """Load and validate worksheet records from a JSON file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc.msg}.") from exc
    if not isinstance(payload, list) or not payload or any(
        not isinstance(it, Mapping) or "sentence" not in it or "target_word" not in it for it in payload
    ):
        raise ValueError(f"Worksheet file {path} must contain a non-empty list of valid records.")
    return [dict(item) for item in payload]


def _simulate_dev_probs(dev_items: Sequence[Mapping[str, Any]], seed: int) -> list[np.ndarray]:
    """Generate deterministic (L, 36) noisy visual probabilities with ~12% CER."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a non-negative integer.")
    rng, all_probs = np.random.default_rng(seed), []
    for item in dev_items:
        if not (word := normalize_word(str(item["target_word"]))):
            raise ValueError("Worksheet target_word must not be empty.")
        probs = np.empty((len(word), _N_CHARS), dtype=np.float32)
        for pos, ch in enumerate(word):
            if (idx := _C2I.get(ch)) is None:
                raise ValueError(f"Unsupported character {ch!r} in target_word {word!r}.")
            row = rng.uniform(*_NOISE["bg"], size=_N_CHARS)
            partners = tuple(p for p in CONFUSION_MAP.get(ch, ()) if p in _C2I and p != ch)
            if rng.random() < _CER:
                pool = partners if (partners and rng.random() < _BIAS) else _OTHERS[ch]
                row[_C2I[pool[int(rng.integers(len(pool)))]]] += float(rng.uniform(*_NOISE["err"]))
                row[idx] += float(rng.uniform(*_NOISE["true"]))
            else:
                row[idx] += float(rng.uniform(*_NOISE["clean"]))
                if partners:
                    row[[_C2I[p] for p in partners]] += float(rng.uniform(*_NOISE["leak"])) / len(partners)
            probs[pos] = (row / row.sum()).astype(np.float32)
        all_probs.append(probs)
    return all_probs


def _save_metrics(
    corrector: ClassicalNLPCorrector, lexicon: Mapping[str, int], dev_items: Sequence[Mapping[str, Any]],
    dev_probs: Sequence[np.ndarray], test_items: Sequence[Mapping[str, Any]], counts: Mapping[str, int],
    ppls: tuple[float, float, float], lambdas: Sequence[float], fusion: Mapping[str, float], saved: Path, out: Path,
) -> dict[str, Any]:
    """Assemble and save evaluation metrics JSON."""
    hits = sum(normalize_word(str(corrector.predict(str(it["sentence"]), p)["predicted_word"]))
               == normalize_word(str(it["target_word"])) for it, p in zip(dev_items, dev_probs, strict=True))
    oov = sum(1 for it in test_items if normalize_word(str(it["target_word"])) not in lexicon)
    bi = ([float(x) for x in raw] if (raw := getattr(corrector, "bigram_lambdas", None)) is not None
          else [float(x) / s if (s := sum(lambdas[1:])) > 0 else d for x, d in zip(lambdas[1:], (0.0, 1.0, 0.0))])
    fb = len(getattr(corrector, "vocab", getattr(corrector, "unigram_counts", lexicon)))
    vocab = (int(rv()) if callable(rv := getattr(corrector, "vocab_size", None)) else
             int(rv) if isinstance(rv, (int, np.integer)) else fb)
    metrics: dict[str, Any] = {
        "ppl_unigram": ppls[0], "ppl_bigram": ppls[1], "ppl_trigram": ppls[2], "lambdas": [float(x) for x in lambdas],
        "bigram_lambdas": bi, "fusion_weights": {str(k): float(v) for k, v in fusion.items()},
        "lexicon_size": len(lexicon), "vocab_size": vocab, "test_oov_rate": float(oov / len(test_items)),
        "dev_simulated_word_acc": float(hits / len(dev_items)),
        "model_size_mb": float(corrector.get_model_size_mb(saved)), "corpus_word_counts": dict(counts),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def train_ngram_pipeline(
    force_download: bool = False, weights_out: Path = DEFAULT_WEIGHTS_PATH,
    metrics_out: Path = _METRICS_PATH, seed: int = _SEED,
) -> dict[str, Any]:
    """Run corpus prep, N-gram training/tuning, evaluation, and artifact saving."""
    prepared, keys = prepare_all(force_download=force_download), ("train_texts", "dev_texts", "ppl_test_texts")
    corpus = next((c for c in (prepared, prepared.get("corpus"))
                   if isinstance(c, Mapping) and all(k in c for k in keys)), None) or prepare_corpus(False)
    train_sents, dev_sents, ppl_test_sents = (_texts_to_sentences(corpus[k]) for k in keys)
    if not train_sents or not dev_sents or not ppl_test_sents:
        raise ValueError("Corpus splits must produce non-empty tokenized sentences.")
    if not (lexicon := build_lexicon([tok for sent in train_sents for tok in sent])):
        raise ValueError("Built lexicon must not be empty.")
    corrector = ClassicalNLPCorrector().train(train_sents, lexicon, k=DEFAULT_SMOOTHING_K)
    lambdas = corrector.tune_lambdas(dev_sents)
    dev_items, test_items = _load_worksheet_items(_DEV_PATH), _load_worksheet_items(_TEST_PATH)
    dev_probs = _simulate_dev_probs(dev_items, seed=seed)
    fusion = corrector.tune_fusion_weights(dev_items, dev_probs)
    ppls = tuple(float(corrector.perplexity(ppl_test_sents, order=o)) for o in (1, 2, 3))
    if not (math.isfinite(ppls[0]) and ppls[0] > ppls[1] > ppls[2] > 0):
        raise RuntimeError(f"Expected ppl_unigram > ppl_bigram > ppl_trigram, got {ppls}.")
    w_path = Path(weights_out)
    w_path.parent.mkdir(parents=True, exist_ok=True)
    saved = Path(corrector.save(w_path) or w_path)
    splits = (("train", train_sents), ("dev", dev_sents), ("ppl_test", ppl_test_sents))
    counts = {n: sum(len(s) for s in ss) for n, ss in splits}
    counts["total"] = sum(counts.values())
    return _save_metrics(corrector, lexicon, dev_items, dev_probs, test_items,
                         counts, ppls, lambdas, fusion, saved, Path(metrics_out))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for training, tuning, and saving the N-gram corrector."""
    p = argparse.ArgumentParser(description="Train and evaluate the classical N-gram corrector.")
    p.add_argument("--force-download", action="store_true", help="Re-download corpus and EMNIST sources.")
    p.add_argument("--weights-out", type=Path, default=DEFAULT_WEIGHTS_PATH, help="Output pickle path.")
    p.add_argument("--metrics-out", type=Path, default=_METRICS_PATH, help="Output metrics JSON path.")
    p.add_argument("--seed", type=int, default=_SEED, help="RNG seed for simulated dev probs.")
    try:
        args = p.parse_args(argv)
        m = train_ngram_pipeline(args.force_download, args.weights_out, args.metrics_out, args.seed)
    except OSError as exc:
        is_dir = exc.errno == errno.EACCES and exc.filename and Path(exc.filename).is_dir()
        msg = _OS_ERRS.get(errno.EISDIR if is_dir else exc.errno, exc.strerror or "I/O failure")
        sys.stderr.write(f"File system error ({exc.filename or 'unspecified path'}): {msg}.\n")
        return 1
    except (ValueError, TypeError, RuntimeError) as exc:
        sys.stderr.write(f"Error: {' '.join(str(exc).splitlines()).strip() or 'N-gram training failed.'}\n")
        return 1
    sys.stdout.write(f"N-gram complete | PPL(1/2/3): {m['ppl_unigram']:.2f}/{m['ppl_bigram']:.2f}/{m['ppl_trigram']:.2f}"
                     f" | Lex: {m['lexicon_size']} | Vocab: {m['vocab_size']} | OOV: {m['test_oov_rate']:.2%}"
                     f" | Dev Acc: {m['dev_simulated_word_acc']:.2%} | Size: {m['model_size_mb']:.2f} MB\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

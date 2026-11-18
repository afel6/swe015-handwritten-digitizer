"""Train, tune, evaluate, and save the classical N-gram language model corrector."""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

if str(_ROOT := Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data._io_errors import describe_os_error
from data.emnist_data import extract_oriented_tensors
from data.prepare_data import (
    EMNIST_ROOT,
    EMNIST_SPLITS_PATH,
    build_lexicon,
    load_emnist_dataset,
    load_emnist_splits,
    prepare_corpus,
    read_data,
    write_data,
)
from data.preprocess import fold_47_to_36_probs
from models.char_models import load_char_cnn_fp32
from models.classical_nlp import (
    DEFAULT_SMOOTHING_K,
    DEFAULT_WEIGHTS_PATH,
    ClassicalNLPCorrector,
    normalize_word,
    tokenize,
)

_METRICS_PATH, _CNN_WEIGHTS = Path("results/ngram_metrics.json"), Path("weights/char_cnn_fp32.pt")
_DEV_PATH, _TEST_PATH = Path("data/worksheets_dev_50.json"), Path("data/worksheets_test_300.json")
_SPLIT_RE = re.compile(r"[.!?\n]+")


def _texts_to_sentences(texts: Sequence[str]) -> list[list[str]]:
    """Split documents on sentence boundaries and tokenize each non-empty sentence."""
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("Each corpus document must be a string.")
    return [toks for text in texts for chunk in _SPLIT_RE.split(text) if (toks := tokenize(chunk))]


def _load_worksheet_items(path: Path) -> list[dict[str, Any]]:
    """Load and validate worksheet records from a JSON file."""
    payload = read_data(Path(path), as_json=True)
    if not isinstance(payload, list) or not payload or any(
        not isinstance(it, Mapping) or "sentence" not in it or "target_word" not in it for it in payload
    ):
        raise ValueError(f"Worksheet file {path} must contain a non-empty list of valid records.")
    return [dict(item) for item in payload]


def compute_dev_cnn_probs(
    dev_items: Sequence[Mapping[str, Any]], cnn_weights: Path = _CNN_WEIGHTS,
    emnist_root: Path = EMNIST_ROOT, splits_path: Path = EMNIST_SPLITS_PATH,
) -> list[np.ndarray]:
    """Run the FP32 CNN on each dev item's real EMNIST validation glyphs; return folded (L, 36) probabilities."""
    model = load_char_cnn_fp32(cnn_weights, device="cpu")
    val_indices = set(load_emnist_splits(splits_path)["val_indices"])
    dataset = load_emnist_dataset(Path(emnist_root), train=True)
    out: list[np.ndarray] = []
    for item in dev_items:
        indices = [int(i) for i in item.get("emnist_indices", [])]
        if item.get("emnist_split") != "val" or not indices or not set(indices) <= val_indices:
            raise ValueError(f"Dev item {item.get('id')} must use EMNIST val indices only.")
        if len(indices) != len(normalize_word(str(item["target_word"]))):
            raise ValueError(f"Dev item {item.get('id')} has {len(indices)} glyphs for target {item['target_word']!r}.")
        glyphs, _ = extract_oriented_tensors(dataset, np.asarray(indices, dtype=np.int64))
        with torch.inference_mode():
            probs_47 = torch.softmax(model(glyphs), dim=1)
        out.append(np.asarray(fold_47_to_36_probs(probs_47).cpu().numpy(), dtype=np.float64))
    return out


def _dev_word_accuracy(corrector: ClassicalNLPCorrector, items: Sequence[Mapping[str, Any]], probs: Sequence[np.ndarray]) -> float:
    hits = sum(normalize_word(str(corrector.predict(str(it["sentence"]), p)["predicted_word"])) == normalize_word(str(it["target_word"]))
               for it, p in zip(items, probs, strict=True))
    return float(hits / len(items))


def train_ngram_pipeline(
    force_download: bool = False, weights_out: Path = DEFAULT_WEIGHTS_PATH,
    metrics_out: Path = _METRICS_PATH, cnn_weights: Path = _CNN_WEIGHTS,
) -> dict[str, Any]:
    """Train the LM on train text, tune on dev text + dev worksheets only, evaluate PPL on held-out text, save artifacts."""
    dev_items = _load_worksheet_items(_DEV_PATH)
    dev_probs = compute_dev_cnn_probs(dev_items, cnn_weights=Path(cnn_weights))
    corpus = prepare_corpus(force_download=force_download)
    train_sents, dev_sents, ppl_test_sents = (_texts_to_sentences(corpus[k]) for k in ("train_texts", "dev_texts", "ppl_test_texts"))
    if not train_sents or not dev_sents or not ppl_test_sents:
        raise ValueError("Corpus splits must produce non-empty tokenized sentences.")
    if not (lexicon := build_lexicon([tok for sent in train_sents for tok in sent])):
        raise ValueError("Built lexicon must not be empty.")
    corrector = ClassicalNLPCorrector().train(train_sents, lexicon, k=DEFAULT_SMOOTHING_K)
    lambdas = corrector.tune_lambdas(dev_sents)
    fusion = corrector.tune_fusion_weights(dev_items, dev_probs)
    ppls = tuple(float(corrector.perplexity(ppl_test_sents, order=o)) for o in (1, 2, 3))
    if not (math.isfinite(ppls[0]) and ppls[0] > ppls[1] > ppls[2] > 0):
        raise RuntimeError(f"Expected ppl_unigram > ppl_bigram > ppl_trigram, got {ppls}.")
    w_path = Path(weights_out)
    w_path.parent.mkdir(parents=True, exist_ok=True)
    saved = corrector.save(w_path)
    test_items = _load_worksheet_items(_TEST_PATH)  # read only for the OOV statistic, never for tuning
    counts = {n: sum(len(s) for s in ss) for n, ss in (("train", train_sents), ("dev", dev_sents), ("ppl_test", ppl_test_sents))}
    metrics: dict[str, Any] = {
        "ppl_unigram": ppls[0], "ppl_bigram": ppls[1], "ppl_trigram": ppls[2],
        "lambdas": [float(x) for x in lambdas], "bigram_lambdas": [float(x) for x in corrector.bigram_lambdas],
        "fusion_weights": {str(k): float(v) for k, v in fusion.items()},
        "lexicon_size": len(corrector.lexicon), "vocab_size": len(corrector.vocab),
        "test_oov_rate": sum(1 for it in test_items if normalize_word(str(it["target_word"])) not in corrector.lexicon) / len(test_items),
        "dev_word_acc_tuned": _dev_word_accuracy(corrector, dev_items, dev_probs),
        "model_size_mb": float(corrector.get_model_size_mb(saved)), "corpus_word_counts": {**counts, "total": sum(counts.values())},
    }
    _write_metrics(Path(metrics_out), metrics)
    return metrics


def _write_metrics(path: Path, metrics: Mapping[str, Any]) -> None:
    """Write metrics JSON; OS errors are mapped to path-naming exceptions."""
    write_data(path, dict(metrics), as_json=True)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for training, tuning, and saving the N-gram corrector."""
    p = argparse.ArgumentParser(description="Train and evaluate the classical N-gram corrector.")
    p.add_argument("--force-download", action="store_true", help="Re-download corpus sources instead of using the cache.")
    p.add_argument("--weights-out", type=Path, default=DEFAULT_WEIGHTS_PATH, help="Output pickle path.")
    p.add_argument("--metrics-out", type=Path, default=_METRICS_PATH, help="Output metrics JSON path.")
    p.add_argument("--cnn-weights", type=Path, default=_CNN_WEIGHTS, help="FP32 CNN checkpoint used for dev visual probabilities.")
    args = p.parse_args(argv)
    try:
        m = train_ngram_pipeline(args.force_download, args.weights_out, args.metrics_out, args.cnn_weights)
    except OSError as exc:
        sys.stderr.write(f"File system error: {describe_os_error(exc)}\n")
        return 1
    except (ValueError, TypeError, RuntimeError) as exc:
        sys.stderr.write(f"Error: {' '.join(str(exc).splitlines()).strip() or 'N-gram training failed.'}\n")
        return 1
    sys.stdout.write(f"N-gram complete | PPL(1/2/3): {m['ppl_unigram']:.2f}/{m['ppl_bigram']:.2f}/{m['ppl_trigram']:.2f}"
                     f" | Lex: {m['lexicon_size']} | Vocab: {m['vocab_size']} | Test OOV: {m['test_oov_rate']:.2%}"
                     f" | Dev Acc (tuning set): {m['dev_word_acc_tuned']:.2%} | Size: {m['model_size_mb']:.2f} MB\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

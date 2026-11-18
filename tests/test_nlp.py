"""Tests for the classical NLP corrector, corpus/worksheet preparation, and n-gram training."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import torch

import data.prepare_data as prep
from data.prepare_data import build_lexicon, load_emnist_splits
from data.preprocess import CANONICAL_36_CHARS
from models.classical_nlp import (
    CONFUSION_MAP,
    MASK_TOKEN,
    ClassicalNLPCorrector,
    extract_cloze_context,
    tokenize,
    unweighted_edit_distance,
    weighted_edit_distance,
)
from scripts import train_ngram

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_PATH = REPO_ROOT / "data" / "emnist_splits.json"
DEV_PATH = REPO_ROOT / "data" / "worksheets_dev_50.json"
TEST_PATH = REPO_ROOT / "data" / "worksheets_test_300.json"
SAMPLES_DIR = REPO_ROOT / "data" / "sample_worksheets"
CORPUS_SOURCES = REPO_ROOT / "results" / "corpus_sources.json"
EMNIST_RAW = REPO_ROOT / "data" / "emnist" / "EMNIST" / "raw"
RAW_CORPUS = REPO_ROOT / "data" / "raw"
CNN_WEIGHTS = REPO_ROOT / "weights" / "char_cnn_fp32.pt"
MAX_SENTENCE_CHARS = 500
_C2I = {ch: i for i, ch in enumerate(CANONICAL_36_CHARS)}

needs_emnist = pytest.mark.skipif(not (EMNIST_RAW / "emnist-balanced-train-labels-idx1-ubyte").is_file(), reason="EMNIST balanced files not downloaded")
needs_corpus = pytest.mark.skipif(not (RAW_CORPUS / "gutenberg").is_dir() or not (RAW_CORPUS / "wikipedia").is_dir(), reason="cached corpus (data/raw) not present")
needs_cnn = pytest.mark.skipif(not CNN_WEIGHTS.is_file(), reason="FP32 CNN weights not present")

TINY_SENTENCES = [
    "STUDENTS LEARN SCIENCE AT SCHOOL", "WE STUDY SCIENCE IN THE LAB", "SCIENCE IS THE STUDY OF NATURE",
    "THE CHILDREN VISITED THE ZOO TODAY", "ANIMALS LIVE IN THE ZOO", "THE WAR ENDED IN 1945",
    "THE SILENCE OF THE NIGHT WAS DEEP", "A BOOK ABOUT SCIENCE", "I READ A BOOK AT SCHOOL",
]


def _sha256(path: Path) -> str:
    """Content hash; text files are compared with LF line endings, as git stores them (checkouts may use CRLF)."""
    raw = path.read_bytes()
    return hashlib.sha256(raw if path.suffix == ".png" else raw.replace(b"\r\n", b"\n")).hexdigest()


def _probs_for(raw: str, top: float = 0.9, second: dict[int, tuple[str, float]] | None = None) -> np.ndarray:
    """Build (L, 36) probabilities whose argmax spells `raw`; optional per-position runner-up."""
    probs = np.zeros((len(raw), len(CANONICAL_36_CHARS)), dtype=np.float64)
    for pos, ch in enumerate(raw):
        alt_ch, alt_p = (second or {}).get(pos, (None, 0.0))
        rest = (1.0 - top - alt_p) / (len(CANONICAL_36_CHARS) - (2 if alt_ch else 1))
        probs[pos, :] = rest
        probs[pos, _C2I[ch]] = top
        if alt_ch:
            probs[pos, _C2I[alt_ch]] = alt_p
    return probs


@pytest.fixture(scope="module")
def tiny_model() -> ClassicalNLPCorrector:
    sents = [tokenize(s) for s in TINY_SENTENCES]
    lexicon = build_lexicon([t for s in sents for t in s], min_count=1)
    return ClassicalNLPCorrector().train(sents, lexicon)


@pytest.fixture(scope="module")
def real_corpus() -> dict[str, Any]:
    if not (RAW_CORPUS / "gutenberg").is_dir():
        pytest.skip("cached corpus (data/raw) not present")
    corpus = prep.prepare_corpus(force_download=False)
    return {k: train_ngram._texts_to_sentences(corpus[k]) for k in ("train_texts", "dev_texts", "ppl_test_texts")}


# ---------------------------------------------------------------- language model


@pytest.mark.parametrize("history", [("<s>", "<s>"), ("<s>", "THE"), ("IN", "THE"), ("THE", "ZOO"), ("QWERTY", "ZXCVB"), ("THE", "NOTAWORD")])
def test_next_word_distribution_sums_to_one(tiny_model: ClassicalNLPCorrector, history: tuple[str, str]) -> None:
    dist = tiny_model.next_word_distribution(history)
    assert dist.shape == (len(tiny_model.vocab),)
    assert np.all(dist > 0.0)
    assert abs(float(dist.sum()) - 1.0) < 1e-6


@pytest.mark.parametrize("history", [("IN", "THE"), ("<s>", "<s>"), ("QWERTY", "THE")])
def test_cond_prob_matches_distribution(tiny_model: ClassicalNLPCorrector, history: tuple[str, str]) -> None:
    dist = tiny_model.next_word_distribution(history)
    for word in ("ZOO", "SCIENCE", "</s>", "UNSEENWORD"):
        wid = tiny_model.word_to_id.get(word, tiny_model.word_to_id["<UNK>"])
        assert tiny_model.cond_prob(word, history) == pytest.approx(float(dist[wid]), rel=1e-9)


def test_seen_trigram_is_more_likely_than_unseen(tiny_model: ClassicalNLPCorrector) -> None:
    assert tiny_model.cond_prob("ZOO", ("IN", "THE")) > tiny_model.cond_prob("NIGHT", ("IN", "THE"))


def _ids(model: ClassicalNLPCorrector, *words: str) -> list[int]:
    return [model.word_to_id[w] for w in words]


def test_higher_order_components_are_unsmoothed_relative_frequencies(tiny_model: ClassicalNLPCorrector) -> None:
    uid, vid, zoo, night = _ids(tiny_model, "IN", "THE", "ZOO", "NIGHT")
    p3, p2, _ = tiny_model._order_probs(uid, vid, zoo)
    assert p3 == tiny_model.trigram_counts[(uid, vid)][zoo] / tiny_model.trigram_totals[(uid, vid)]
    assert p2 == tiny_model.bigram_counts[vid][zoo] / tiny_model.bigram_totals[vid]
    p3_unseen, _, _ = tiny_model._order_probs(uid, vid, night)
    assert p3_unseen == 0.0  # "IN THE NIGHT" never occurs; interpolation, not add-k, supplies its mass


def test_unseen_contexts_back_off_to_the_lower_order(tiny_model: ClassicalNLPCorrector) -> None:
    unk = tiny_model.word_to_id["<UNK>"]
    for wid in _ids(tiny_model, "ZOO", "SCIENCE"):
        p3, p2, _ = tiny_model._order_probs(unk, tiny_model.word_to_id["THE"], wid)
        assert p3 == p2  # trigram context (UNK, THE) unseen -> bigram estimate
        q3, q2, q1 = tiny_model._order_probs(unk, unk, wid)
        assert q3 == q2 == q1  # both contexts unseen -> unigram estimate


def test_perplexity_matches_cond_prob_definition(tiny_model: ClassicalNLPCorrector) -> None:
    held_out = [tokenize("THE CHILDREN STUDY SCIENCE"), tokenize("ANIMALS LIVE IN THE LAB")]
    logs = []
    for sent in held_out:
        padded = ["<s>", "<s>", *sent, "</s>"]
        logs += [math.log(tiny_model.cond_prob(padded[i], (padded[i - 2], padded[i - 1]))) for i in range(2, len(padded))]
    assert tiny_model.perplexity(held_out, order=3) == pytest.approx(math.exp(-sum(logs) / len(logs)), rel=1e-9)


def test_perplexity_decreases_with_order_on_held_out_text(real_corpus: dict[str, Any]) -> None:
    train, dev, held_out = real_corpus["train_texts"], real_corpus["dev_texts"], real_corpus["ppl_test_texts"]
    model = ClassicalNLPCorrector().train(train, build_lexicon([t for s in train for t in s]))
    model.tune_lambdas(dev)
    ppl = [model.perplexity(held_out, order=o) for o in (1, 2, 3)]
    assert all(math.isfinite(p) for p in ppl)
    assert ppl[0] > ppl[1] > ppl[2], ppl


def test_tune_lambdas_returns_valid_weights(tiny_model: ClassicalNLPCorrector) -> None:
    model = ClassicalNLPCorrector().train([tokenize(s) for s in TINY_SENTENCES], tiny_model.lexicon)
    lambdas = model.tune_lambdas([tokenize("THE CHILDREN LEARN SCIENCE AT THE ZOO")])
    assert len(lambdas) == 4 and abs(sum(lambdas) - 1.0) < 1e-5 and all(x >= 0.0 for x in lambdas)
    assert model.bigram_lambdas[0] == 0.0 and abs(sum(model.bigram_lambdas) - 1.0) < 1e-5


def _markov_sentences(rng: np.random.Generator, table: dict[tuple[str, str], list[str]], words: list[str], count: int) -> list[list[str]]:
    sents = []
    for _ in range(count):
        a, b = words[int(rng.integers(len(words)))], words[int(rng.integers(len(words)))]
        sent = [a, b]
        for _ in range(8):
            options = table[(sent[-2], sent[-1])]
            sent.append(options[int(rng.integers(len(options)))])
        sents.append(sent)
    return sents


def test_perplexity_decreases_with_order_on_synthetic_second_order_text() -> None:
    rng = np.random.default_rng(7)
    words = [a + b for a in "ABCDEF" for b in "GHIJ"]  # 24 two-letter words, all canonical lexicon entries
    table = {(a, b): [words[int(i)] for i in rng.choice(len(words), size=2, replace=False)] for a in words for b in words}
    train, dev, held_out = (_markov_sentences(rng, table, words, n) for n in (6000, 300, 300))
    model = ClassicalNLPCorrector().train(train, build_lexicon([t for s in train for t in s], min_count=1))
    model.tune_lambdas(dev)
    ppl = [model.perplexity(held_out, order=o) for o in (1, 2, 3)]
    assert ppl[0] > ppl[1] > ppl[2], ppl


@pytest.mark.parametrize("field", ["lambdas", "bigram_lambdas", "unigram_lambdas"])
@pytest.mark.parametrize("bad", [(0.5, 0.5, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)])
def test_lambdas_without_unigram_or_uniform_mass_are_rejected(tiny_model: ClassicalNLPCorrector, bad: tuple[float, ...], field: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="l1"):
        tiny_model.next_word_distribution(("IN", "THE"), lambdas=bad)
    path = tiny_model.save(tmp_path / "lm.pkl")
    import pickle

    payload = pickle.loads(path.read_bytes())
    path.write_bytes(pickle.dumps({**payload, field: bad}))
    with pytest.raises(ValueError, match="l1"):
        ClassicalNLPCorrector.load(path)


@pytest.mark.parametrize("ok", [(0.5, 0.0, 0.5, 0.0), (0.5, 0.5, 0.0, 1e-9), (0.0, 0.0, 0.0, 1.0)])
def test_lambdas_with_some_unigram_or_uniform_mass_are_accepted(tiny_model: ClassicalNLPCorrector, ok: tuple[float, ...]) -> None:
    dist = tiny_model.next_word_distribution(("IN", "THE"), lambdas=ok)
    assert abs(float(dist.sum()) - 1.0) < 1e-6 and np.all(dist > 0.0)


# ---------------------------------------------------------------- correction


def test_predict_corrects_digit_confusion_in_context(tiny_model: ClassicalNLPCorrector) -> None:
    out = tiny_model.predict("Students learn [MASK] at school.", _probs_for("SC1ENCE"))
    assert out["raw_word"] == "SC1ENCE"
    assert out["predicted_word"] == "SCIENCE"
    assert out["numeric_preserved"] is False


def test_numeric_guard_keeps_confident_year(tiny_model: ClassicalNLPCorrector) -> None:
    out = tiny_model.predict("The war ended in [MASK].", _probs_for("1945", top=0.95))
    assert out["numeric_preserved"] is True
    assert out["predicted_word"] == "1945"


def test_numeric_guard_does_not_preserve_low_confidence_digits(tiny_model: ClassicalNLPCorrector) -> None:
    probs = _probs_for("200", top=0.45, second={0: ("Z", 0.40), 1: ("O", 0.40), 2: ("O", 0.40)})
    out = tiny_model.predict("Animals live in the [MASK].", probs)
    assert out["raw_word"] == "200"
    assert out["numeric_preserved"] is False
    assert out["predicted_word"] == "ZOO"


def test_generate_candidates_respects_limit(tiny_model: ClassicalNLPCorrector) -> None:
    cands, meta = tiny_model.generate_candidates(torch.from_numpy(_probs_for("SC1ENCE")), max_candidates=2)
    assert 1 <= len(cands) <= 2 and meta["raw_word"] == "SC1ENCE"
    with pytest.raises(ValueError):
        tiny_model.generate_candidates(np.ones((3, 35)))


def test_tune_fusion_weights_uses_given_items(tiny_model: ClassicalNLPCorrector) -> None:
    items = [{"sentence": "Students learn [MASK] at school.", "target_word": "SCIENCE"}, {"sentence": "Animals live in the [MASK].", "target_word": "ZOO"}]
    weights = tiny_model.tune_fusion_weights(items, [_probs_for("SC1ENCE"), _probs_for("Z00", top=0.6)])
    assert set(weights) == {"vis_weight", "edit_penalty", "lm_weight"}
    assert sum(weights.values()) == pytest.approx(1.0) and min(weights.values()) > 0.0
    with pytest.raises(ValueError):
        tiny_model.tune_fusion_weights(items, [_probs_for("SC1ENCE")])


def test_candidate_pool_does_not_depend_on_fusion_weights(tiny_model: ClassicalNLPCorrector, monkeypatch: pytest.MonkeyPatch) -> None:
    probs = _probs_for("SC1ENCE", top=0.55, second={2: ("L", 0.40), 4: ("M", 0.40)})
    pools = []
    for weights in ({"vis_weight": 0.9, "edit_penalty": 0.05, "lm_weight": 0.05}, {"vis_weight": 0.05, "edit_penalty": 0.9, "lm_weight": 0.05}):
        monkeypatch.setattr(tiny_model, "fusion_weights", weights)
        pools.append([c["word"] for c in tiny_model.generate_candidates(probs, max_candidates=1)[0]])
    assert pools[0] == pools[1] and len(pools[0]) == 1


def test_untrained_model_refuses_use() -> None:
    with pytest.raises(RuntimeError):
        ClassicalNLPCorrector().predict("A [MASK] b", _probs_for("AB"))


def test_numeric_guard_keeps_confident_digits_near_a_lexicon_word(tiny_model: ClassicalNLPCorrector) -> None:
    probs = _probs_for("200", top=0.95, second={0: ("Z", 0.03), 1: ("O", 0.03), 2: ("O", 0.03)})
    out = tiny_model.predict("Animals live in the [MASK].", probs)
    assert out["numeric_preserved"] is True
    assert out["predicted_word"] == "200"


def test_fusion_tuning_changes_weights_when_defaults_pick_the_wrong_word() -> None:
    sents = [tokenize(s) for s in TINY_SENTENCES]
    model = ClassicalNLPCorrector().train(sents, build_lexicon([t for s in sents for t in s], min_count=1))
    probs = _probs_for("SILENCE", top=0.9)
    for pos, (read, alt) in {1: ("I", "C"), 2: ("L", "I")}.items():  # clear SILENCE read, SCIENCE letters as runners-up
        probs[pos, :] = 0.05 / 34
        probs[pos, _C2I[read]] = 0.6
        probs[pos, _C2I[alt]] = 0.35
    sentence = "Students learn [MASK] at school."
    assert model.predict(sentence, probs)["predicted_word"] == "SCIENCE"  # the context outweighs the clear SILENCE glyphs
    tuned = model.tune_fusion_weights([{"sentence": sentence, "target_word": "SILENCE"}], [probs])
    assert tuned["lm_weight"] < 0.25  # the defaults give the LM a 0.25 share of the (normalised) weights; tuning must move off it
    assert model.predict(sentence, probs)["predicted_word"] == "SILENCE"


# ---------------------------------------------------------------- cloze parsing


@pytest.mark.parametrize("sentence", ["The ___ is hot.", "The _____ is hot.", "The [MASK] is hot.", "The [mask] is hot.", "The [Mask] is hot."])
def test_cloze_blank_variants_normalise_to_one_mask(sentence: str) -> None:
    assert extract_cloze_context(sentence) == (["THE"], ["IS", "HOT"])


@pytest.mark.parametrize("sentence", ["No blank here.", "Two [MASK] blanks [mask] here.", "Two ___ blanks ___ here.", "Mixed ___ and [MASK]."])
def test_cloze_requires_exactly_one_mask(sentence: str) -> None:
    with pytest.raises(ValueError):
        extract_cloze_context(sentence)


def test_cloze_sentence_length_limit() -> None:
    prefix = "A [MASK] "
    ok = prefix + "x" * (MAX_SENTENCE_CHARS - len(prefix))
    assert len(ok) == MAX_SENTENCE_CHARS
    assert extract_cloze_context(ok)[0] == ["A"]
    with pytest.raises(ValueError):
        extract_cloze_context(ok + "x")


def test_cloze_mixed_case_mask_splits_context_correctly() -> None:
    assert extract_cloze_context("Paris is in [mAsK] today") == (["PARIS", "IS", "IN"], ["TODAY"])


def test_cloze_two_underscores_are_not_a_blank() -> None:
    with pytest.raises(ValueError):
        extract_cloze_context("The __ is hot.")


# ---------------------------------------------------------------- edit distance


@pytest.mark.parametrize(("a", "b", "expected"), [("KITTEN", "SITTING", 3), ("", "ABC", 3), ("ABC", "ABC", 0), ("FLAW", "LAWN", 2), ("SC1ENCE", "SCIENCE", 1)])
def test_unweighted_edit_distance_exact(a: str, b: str, expected: int) -> None:
    assert unweighted_edit_distance(a, b) == expected
    assert unweighted_edit_distance(b, a) == expected


@pytest.mark.parametrize(("a", "b"), [("0", "O"), ("1", "I"), ("1", "L"), ("5", "S"), ("8", "B"), ("2", "Z")])
def test_weighted_edit_distance_cheaper_for_confusion_pairs(a: str, b: str) -> None:
    assert b in CONFUSION_MAP[a]
    assert weighted_edit_distance(a, b) < weighted_edit_distance(a, "X")
    assert weighted_edit_distance("SC" + a + "ENCE", "SC" + b + "ENCE") < unweighted_edit_distance("SC" + a + "ENCE", "SC" + b + "ENCE")


def test_weighted_edit_distance_identity_is_zero() -> None:
    assert weighted_edit_distance("SCIENCE", "SCIENCE") == 0.0
    with pytest.raises(TypeError):
        weighted_edit_distance("A", 1)  # type: ignore[arg-type]


# ---------------------------------------------------------------- file loading errors


def _pretend_paths_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "exists", lambda self, **kw: True)
    monkeypatch.setattr(Path, "is_file", lambda self, **kw: True)


@pytest.mark.parametrize("loader", [ClassicalNLPCorrector.load, load_emnist_splits])
def test_loaders_map_missing_path_to_file_not_found(loader: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "nope" / "model.bin"
    _pretend_paths_exist(monkeypatch)  # detection must come from the OS error (ENOENT), not exists()/is_file()
    with pytest.raises(FileNotFoundError) as info:
        loader(missing)
    assert info.value.errno == errno.ENOENT
    assert str(missing) in str(info.value) and info.value.filename == str(missing)


@pytest.mark.parametrize("loader", [ClassicalNLPCorrector.load, load_emnist_splits])
def test_loaders_map_directory_to_is_a_directory(loader: Any, tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError) as info:
        loader(tmp_path)
    assert str(tmp_path) in str(info.value) and info.value.filename == str(tmp_path)


def test_load_rejects_corrupt_model_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match=r"bad\.pkl"):
        ClassicalNLPCorrector.load(bad)


def test_save_load_round_trip_gives_identical_predictions(tiny_model: ClassicalNLPCorrector, tmp_path: Path) -> None:
    saved = tiny_model.save(tmp_path / "lm.pkl")
    loaded = ClassicalNLPCorrector.load(saved)
    for sentence, raw in (("Students learn [MASK] at school.", "SC1ENCE"), ("Animals live in the [MASK].", "Z00")):
        assert loaded.predict(sentence, _probs_for(raw)) == tiny_model.predict(sentence, _probs_for(raw))
    assert np.array_equal(loaded.next_word_distribution(("IN", "THE")), tiny_model.next_word_distribution(("IN", "THE")))
    assert loaded.get_model_size_mb() == pytest.approx(saved.stat().st_size / (1024 * 1024))


def test_worksheet_and_metrics_io_report_directories_with_the_path(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError) as read_err:
        prep.load_worksheet_items(tmp_path)
    assert read_err.value.filename == str(tmp_path)
    with pytest.raises(IsADirectoryError) as write_err:
        train_ngram._write_metrics(tmp_path, {"ppl_unigram": 1.0})
    assert write_err.value.filename == str(tmp_path)


def test_checkpoint_loaders_name_the_missing_path(tmp_path: Path) -> None:
    from models.char_models import load_char_cnn_fp32
    from models.quantization import load_quantized_cnn

    missing = tmp_path / "missing.pt"
    for loader in (load_char_cnn_fp32, load_quantized_cnn):
        with pytest.raises(FileNotFoundError) as info:
            loader(missing)
        assert info.value.filename == str(missing) and str(missing) in str(info.value)


# ---------------------------------------------------------------- split leak guards


@pytest.fixture(scope="module")
def emnist_train_targets() -> np.ndarray:
    if not (EMNIST_RAW / "emnist-balanced-train-labels-idx1-ubyte").is_file():
        pytest.skip("EMNIST balanced files not downloaded")
    from torchvision.datasets import EMNIST

    return EMNIST(str(REPO_ROOT / "data" / "emnist"), split="balanced", train=True, download=False).targets.numpy()


def _cnn_split(targets: np.ndarray) -> tuple[set[int], set[int]]:
    from scripts.train_vision import compute_stratified_split

    train_idx, val_idx = compute_stratified_split(targets)
    return set(train_idx.tolist()), set(val_idx.tolist())


def test_splits_file_matches_cnn_training_split(emnist_train_targets: np.ndarray) -> None:
    cnn_train, cnn_val = _cnn_split(emnist_train_targets)
    splits = load_emnist_splits(SPLITS_PATH)
    assert set(splits["train_indices"]) == cnn_train
    assert set(splits["val_indices"]) == cnn_val


def test_build_emnist_splits_reproduces_cnn_split(emnist_train_targets: np.ndarray, tmp_path: Path) -> None:
    out = tmp_path / "splits.json"
    built = prep.build_emnist_splits(root=REPO_ROOT / "data" / "emnist", out_path=out)
    cnn_train, cnn_val = _cnn_split(emnist_train_targets)
    assert set(built["train_indices"]) == cnn_train and set(built["val_indices"]) == cnn_val
    assert _sha256(out) == _sha256(SPLITS_PATH)


def test_dev_worksheet_characters_come_only_from_cnn_validation(emnist_train_targets: np.ndarray) -> None:
    cnn_train, cnn_val = _cnn_split(emnist_train_targets)
    dev = json.loads(DEV_PATH.read_text(encoding="utf-8"))
    indices = [i for it in dev for i in it["emnist_indices"]]
    assert len(dev) == 50 and all(it["emnist_split"] == "val" for it in dev)
    assert set(indices) <= cnn_val
    assert not set(indices) & cnn_train
    for it in dev:
        assert [CANONICAL_36_CHARS[int(emnist_train_targets[i])] for i in it["emnist_indices"]] == list(it["target_word"])


def test_test_worksheets_use_emnist_test_set_and_are_disjoint_from_dev() -> None:
    dev = json.loads(DEV_PATH.read_text(encoding="utf-8"))
    test = json.loads(TEST_PATH.read_text(encoding="utf-8"))
    assert len(test) == 300 and all(it["emnist_split"] == "test" for it in test)
    assert not {it["id"] for it in dev} & {it["id"] for it in test}
    assert not {it["sentence"] for it in dev} & {it["sentence"] for it in test}
    assert all(it["sentence"].count(MASK_TOKEN) == 1 for it in dev + test)


def test_split_helpers_live_in_data_and_are_shared() -> None:
    from data import emnist_data
    from scripts import train_vision

    assert train_vision.compute_stratified_split is emnist_data.compute_stratified_split
    assert train_vision.extract_oriented_tensors is emnist_data.extract_oriented_tensors
    proc = subprocess.run([sys.executable, "-c", "import sys, data.prepare_data, scripts.train_ngram; print('scripts.train_vision' in sys.modules)"],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, check=True)
    assert proc.stdout.strip() == "False"  # data/ must not depend on the training script


def test_split_constants_match_the_cnn_training_script() -> None:
    import inspect

    from data.emnist_data import SPLIT_SEED, TRAIN_FRAC
    from scripts.train_vision import SEED

    assert SEED == SPLIT_SEED and TRAIN_FRAC == pytest.approx(0.9)
    assert "seed" not in inspect.signature(prep.build_emnist_splits).parameters  # the split is fixed, not configurable


# ---------------------------------------------------------------- frozen data and corpus


def _forbid_reading(monkeypatch: pytest.MonkeyPatch, filename: str) -> None:
    """Fail the test if any code opens a file with this name (read_bytes, read_text or open)."""
    import builtins

    real_read_bytes, real_read_text, real_open = Path.read_bytes, Path.read_text, builtins.open

    def check(target: Any) -> None:
        if isinstance(target, (str, os.PathLike)) and Path(target).name == filename:
            raise AssertionError(f"{filename} must not be read here")

    monkeypatch.setattr(Path, "read_bytes", lambda self: (check(self), real_read_bytes(self))[1])
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (check(self), real_read_text(self, *a, **k))[1])
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (check(f), real_open(f, *a, **k))[1])


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(httpx.Client, "send", refuse)


@needs_corpus
def test_prepare_corpus_does_not_write_corpus_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _block_network(monkeypatch)
    target = tmp_path / "corpus_sources.json"
    monkeypatch.setattr(prep, "CORPUS_SOURCES_PATH", target)
    corpus = prep.prepare_corpus(force_download=False)
    assert not target.exists()
    assert corpus["stats"]["word_counts"]["train"] >= prep.TRAIN_MIN_WORDS


@needs_corpus
@needs_emnist
def test_prepare_all_reproduces_frozen_files_byte_for_byte(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _block_network(monkeypatch)
    outs = {"EMNIST_SPLITS_PATH": tmp_path / "splits.json", "DEV_WORKSHEETS_PATH": tmp_path / "dev.json", "TEST_WORKSHEETS_PATH": tmp_path / "test.json",
            "SAMPLE_WORKSHEETS_DIR": tmp_path / "samples", "CORPUS_SOURCES_PATH": tmp_path / "corpus_sources.json"}
    for name, path in outs.items():
        monkeypatch.setattr(prep, name, path)
    prep.prepare_all(force_download=False)
    assert _sha256(outs["TEST_WORKSHEETS_PATH"]) == _sha256(TEST_PATH)
    assert _sha256(outs["DEV_WORKSHEETS_PATH"]) == _sha256(DEV_PATH)
    assert _sha256(outs["EMNIST_SPLITS_PATH"]) == _sha256(SPLITS_PATH)
    assert _sha256(outs["CORPUS_SOURCES_PATH"]) == _sha256(CORPUS_SOURCES)
    committed = sorted(p.name for p in SAMPLES_DIR.iterdir())
    assert sorted(p.name for p in outs["SAMPLE_WORKSHEETS_DIR"].iterdir()) == committed
    for name in committed:
        assert _sha256(outs["SAMPLE_WORKSHEETS_DIR"] / name) == _sha256(SAMPLES_DIR / name), name


def _mock_client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_gutenberg_book_strips_boilerplate_and_caches(tmp_path: Path) -> None:
    body = "header\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nCall me Ishmael.\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nfooter"
    with _mock_client(lambda req: httpx.Response(200, text=body)) as client:
        text, url = prep._fetch_gutenberg_book(client, 2701, tmp_path, force_download=True)
    assert text == "Call me Ishmael." and url.endswith("pg2701.txt")
    assert (tmp_path / "pg2701.txt").is_file()
    with _mock_client(lambda req: httpx.Response(500)) as client:
        assert prep._fetch_gutenberg_book(client, 2701, tmp_path, force_download=False)[0] == "Call me Ishmael."
        with pytest.raises(RuntimeError, match="2701"):
            prep._fetch_gutenberg_book(client, 2701, tmp_path, force_download=True)


def test_fetch_wiki_article_parses_extract_and_reports_missing(tmp_path: Path) -> None:
    page = {"query": {"pages": {"1": {"title": "Moon", "extract": "== Intro ==\nThe Moon orbits Earth."}}}}
    with _mock_client(lambda req: httpx.Response(200, json=page)) as client:
        doc = prep._fetch_wiki_article(client, "simple", "Moon", tmp_path, force_download=True)
    assert doc["text"] == "The Moon orbits Earth." and doc["url"].endswith("/wiki/Moon")
    missing = {"query": {"pages": {"-1": {"title": "Nope", "missing": ""}}}}
    with _mock_client(lambda req: httpx.Response(200, json=missing)) as client, pytest.raises(RuntimeError, match="Nope"):
        prep._fetch_wiki_article(client, "simple", "Nope", tmp_path, force_download=True)


def test_download_emnist_rejects_checksum_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_client = httpx.Client
    monkeypatch.setattr(prep.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=b"junk")), **{k: v for k, v in kw.items() if k != "verify"}))
    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        prep._download_emnist_balanced(tmp_path)


def test_build_lexicon_counts_only_canonical_words() -> None:
    lex = build_lexicon(["THE", "THE", "CAT", "1945", "A", "X", "THE"], min_count=1)
    assert lex == {"THE": 3, "A": 1, "CAT": 1}
    with pytest.raises(TypeError):
        build_lexicon("THE CAT")  # type: ignore[arg-type]


def test_wiki_cache_missing_fields_is_refetched(tmp_path: Path) -> None:
    (tmp_path / "simple__moon.json").write_text(json.dumps({"text": "Old text."}), encoding="utf-8")
    page = {"query": {"pages": {"1": {"title": "Moon", "extract": "The Moon orbits Earth."}}}}
    with _mock_client(lambda req: httpx.Response(200, json=page)) as client:
        doc = prep._fetch_wiki_article(client, "simple", "Moon", tmp_path, force_download=False)
    assert doc["title"] == "Moon" and doc["text"] == "The Moon orbits Earth."


# ---------------------------------------------------------------- training pipeline


@needs_emnist
@needs_cnn
def test_dev_probs_come_from_real_cnn_images() -> None:
    dev = json.loads(DEV_PATH.read_text(encoding="utf-8"))
    probs = train_ngram.compute_dev_cnn_probs(dev, cnn_weights=CNN_WEIGHTS, emnist_root=REPO_ROOT / "data" / "emnist")
    assert len(probs) == len(dev)
    hits = total = 0
    for item, p in zip(dev, probs, strict=True):
        assert p.shape == (len(item["target_word"]), 36) and np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
        hits += sum(CANONICAL_36_CHARS[int(i)] == ch for i, ch in zip(p.argmax(axis=1), item["target_word"]))
        total += len(item["target_word"])
    assert hits / total > 0.8  # the CNN reads the real dev glyphs: most argmax characters match the target word
    with pytest.raises(ValueError, match="val"):
        train_ngram.compute_dev_cnn_probs([{**dev[0], "emnist_split": "test"}], cnn_weights=CNN_WEIGHTS, emnist_root=REPO_ROOT / "data" / "emnist")


@needs_corpus
@needs_emnist
@needs_cnn
def test_train_pipeline_writes_metrics_without_touching_frozen_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _block_network(monkeypatch)
    frozen = [TEST_PATH, DEV_PATH, SPLITS_PATH, CORPUS_SOURCES, *sorted(SAMPLES_DIR.iterdir())]
    before = {p: _sha256(p) for p in frozen}
    weights, metrics_path = tmp_path / "lm.pkl", tmp_path / "metrics.json"
    _forbid_reading(monkeypatch, TEST_PATH.name)
    metrics = train_ngram.train_ngram_pipeline(weights_out=weights, metrics_out=metrics_path, cnn_weights=CNN_WEIGHTS)
    monkeypatch.undo()
    assert {p: _sha256(p) for p in frozen} == before
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == metrics
    required = {"ppl_unigram", "ppl_bigram", "ppl_trigram", "lambdas", "fusion_weights", "lexicon_size", "vocab_size", "model_size_mb", "corpus_word_counts"}
    assert required <= set(metrics) and "test_oov_rate" not in metrics  # the test set is never read by the training pipeline
    assert metrics["ppl_unigram"] > metrics["ppl_bigram"] > metrics["ppl_trigram"] > 0
    assert len(metrics["lambdas"]) == 4 and abs(sum(metrics["lambdas"]) - 1.0) < 1e-5
    model = ClassicalNLPCorrector.load(weights)
    assert metrics["vocab_size"] == len(model.vocab) and metrics["lexicon_size"] == len(model.lexicon)
    assert metrics["model_size_mb"] == pytest.approx(weights.stat().st_size / (1024 * 1024))


def test_non_object_worksheet_item_is_a_type_error(tmp_path: Path) -> None:
    path = tmp_path / "ws.json"
    path.write_text(json.dumps(["just a string"]), encoding="utf-8")
    with pytest.raises(TypeError, match="not an object"):
        prep.load_worksheet_items(path)


def test_worksheet_and_corpus_inputs_are_validated(tmp_path: Path) -> None:
    bad_json, empty = tmp_path / "bad.json", tmp_path / "empty.json"
    bad_json.write_text("{", encoding="utf-8")
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        prep.load_worksheet_items(bad_json)
    with pytest.raises(ValueError, match="non-empty list"):
        prep.load_worksheet_items(empty)
    with pytest.raises(TypeError):
        train_ngram._texts_to_sentences(["ok", 3])  # type: ignore[list-item]
    assert train_ngram._texts_to_sentences(["One two. Three!\nFour"]) == [["ONE", "TWO"], ["THREE"], ["FOUR"]]


@needs_emnist
@needs_cnn
def test_dev_probs_reject_glyph_count_mismatch() -> None:
    item = json.loads(DEV_PATH.read_text(encoding="utf-8"))[0]
    with pytest.raises(ValueError, match="glyphs"):
        train_ngram.compute_dev_cnn_probs([{**item, "emnist_indices": item["emnist_indices"][:-1]}], cnn_weights=CNN_WEIGHTS, emnist_root=REPO_ROOT / "data" / "emnist")


@needs_emnist
@needs_cnn
def test_dev_probs_reject_out_of_range_index(tmp_path: Path) -> None:
    item = json.loads(DEV_PATH.read_text(encoding="utf-8"))[0]
    splits = load_emnist_splits(SPLITS_PATH)
    bad_index = 10**9
    fake_splits = tmp_path / "splits.json"
    fake_splits.write_text(json.dumps({**splits, "val_indices": [*splits["val_indices"], bad_index]}), encoding="utf-8")
    bad = {**item, "emnist_indices": [*item["emnist_indices"][:-1], bad_index]}
    with pytest.raises(ValueError, match="dataset bounds"):
        train_ngram.compute_dev_cnn_probs([bad], cnn_weights=CNN_WEIGHTS, emnist_root=REPO_ROOT / "data" / "emnist", splits_path=fake_splits)


# ---------------------------------------------------------------- command-line interfaces


def test_prepare_data_script_run_by_path_reports_errors_in_one_line(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "emnist_root_is_a_file"
    not_a_dir.write_text("x", encoding="utf-8")
    proc = subprocess.run([sys.executable, "-W", "error", str(REPO_ROOT / "data" / "prepare_data.py"), "--emnist-root", str(not_a_dir)],
                          cwd=tmp_path, capture_output=True, text=True, timeout=300, check=False)
    assert proc.returncode == 1, proc.stderr
    lines = proc.stderr.strip().splitlines()
    assert len(lines) == 1 and "Traceback" not in proc.stderr and str(not_a_dir) in lines[0]


def test_train_ngram_cli_reports_missing_file_in_one_line(tmp_path: Path) -> None:
    missing = tmp_path / "missing_cnn.pt"
    proc = subprocess.run([sys.executable, "-W", "error", str(REPO_ROOT / "scripts" / "train_ngram.py"), "--cnn-weights", str(missing),
                           "--weights-out", str(tmp_path / "lm.pkl"), "--metrics-out", str(tmp_path / "m.json")],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, check=False)
    assert proc.returncode == 1, proc.stderr
    lines = proc.stderr.strip().splitlines()
    assert len(lines) == 1 and "Traceback" not in proc.stderr
    assert str(missing) in lines[0]
    assert not (tmp_path / "lm.pkl").exists()


def test_prepare_data_cli_reports_os_error_in_one_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "gone.json"

    def failing_prepare_all(force_download: bool = False, emnist_root: Path = prep.EMNIST_ROOT) -> dict[str, Any]:
        return prep.read_data(missing, as_json=True)

    monkeypatch.setattr(prep, "prepare_all", failing_prepare_all)
    assert prep.main([]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and str(missing) in err[0] and "Traceback" not in err[0]


def test_train_ngram_main_prints_summary_and_maps_errors(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    fake = {"ppl_unigram": 9.0, "ppl_bigram": 5.0, "ppl_trigram": 4.0, "lexicon_size": 3, "vocab_size": 5, "dev_word_acc_tuned": 0.5, "model_size_mb": 1.0}
    monkeypatch.setattr(train_ngram, "train_ngram_pipeline", lambda *a, **k: fake)
    assert train_ngram.main([]) == 0
    assert "PPL(1/2/3): 9.00/5.00/4.00" in capsys.readouterr().out

    def raise_plain_os_error(*_a: Any, **_k: Any) -> Any:
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(tmp_path / "x.json"))

    monkeypatch.setattr(train_ngram, "train_ngram_pipeline", raise_plain_os_error)
    assert train_ngram.main([]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and str(tmp_path / "x.json") in err[0]

    def raise_value_error(*_a: Any, **_k: Any) -> Any:
        raise ValueError("bad\nworksheet")

    monkeypatch.setattr(train_ngram, "train_ngram_pipeline", raise_value_error)
    assert train_ngram.main([]) == 1
    assert capsys.readouterr().err == "Error: bad worksheet\n"


def test_prepare_data_cli_keeps_path_for_unmapped_os_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    locked = tmp_path / "emnist-balanced-train-images-idx3-ubyte"

    def locked_file(force_download: bool = False, emnist_root: Path = prep.EMNIST_ROOT) -> dict[str, Any]:
        raise PermissionError(errno.EACCES, "The process cannot access the file", str(locked))

    monkeypatch.setattr(prep, "prepare_all", locked_file)
    assert prep.main([]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and str(locked) in err[0]


def test_model_file_cannot_run_code_when_loaded(tmp_path: Path) -> None:
    import pickle

    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self) -> tuple[Any, tuple[str]]:
            return (os.mkdir, (str(marker),))

    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps({"vocab": Evil()}))
    with pytest.raises(ValueError) as info:
        ClassicalNLPCorrector.load(path)
    assert not marker.exists() and "\\n" not in str(info.value) and "evil.pkl" in str(info.value)


def test_released_model_loads_through_the_restricted_unpickler(tiny_model: ClassicalNLPCorrector, tmp_path: Path) -> None:
    loaded = ClassicalNLPCorrector.load(tiny_model.save(tmp_path / "lm.pkl"))
    assert loaded.vocab == tiny_model.vocab

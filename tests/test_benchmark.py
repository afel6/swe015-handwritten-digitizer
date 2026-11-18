"""End-to-end tests of the benchmark and BERT-tuning scripts with tiny models (no network, no test-set pass)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

import scripts.evaluate_all as evaluate_all
import scripts.train_ngram as train_ngram
import scripts.tune_bert_fusion as tune_bert
from data.preprocess import CANONICAL_36_CHARS
from models.bert_reranker import BertMLMReranker
from models.classical_nlp import ClassicalNLPCorrector, tokenize

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_PATH = REPO_ROOT / "data" / "worksheets_dev_50.json"
WEIGHTS = REPO_ROOT / "weights"
needs_assets = pytest.mark.skipif(not (REPO_ROOT / "data" / "emnist" / "EMNIST" / "raw" / "emnist-balanced-test-labels-idx1-ubyte").is_file()
                                  or not (WEIGHTS / "char_cnn_int8.pt").is_file(), reason="EMNIST files or vision weights missing")
_PARAMS = {"alpha": 1.0, "lambda_vis": 1.0, "lambda_bert": 1.0, "gamma": 1.0}


def _tiny_reranker(*_a: Any, **_k: Any) -> BertMLMReranker:
    torch.manual_seed(1)
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "sun", "is", "a", "##b", "##c"]
    config = BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=256)
    return BertMLMReranker("tiny", model=BertForMaskedLM(config).eval(), tokenizer=BertTokenizer(vocab={w: i for i, w in enumerate(words)}))


def _dev_items(n: int) -> list[dict[str, Any]]:
    return json.loads(DEV_PATH.read_text(encoding="utf-8"))[:n]


def _tiny_lm(items: list[dict[str, Any]]) -> ClassicalNLPCorrector:
    sents = [tokenize(it["sentence"].replace("[MASK]", it["target_word"])) for it in items] + [tokenize("THE SUN IS HOT")]
    return ClassicalNLPCorrector().train(sents, {w: 1 for s in sents for w in s if w.isalpha() and len(w) > 1})


def _onehot_probs(word: str) -> np.ndarray:
    probs = np.full((len(word), 36), 0.1 / 35)
    for i, ch in enumerate(word):
        probs[i, CANONICAL_36_CHARS.index(ch)] = 0.9
    return probs


@needs_assets
def test_evaluate_writes_all_outputs_with_tiny_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    items = _dev_items(3)
    worksheets = tmp_path / "ws.json"
    worksheets.write_text(json.dumps(items), encoding="utf-8")
    weights = {k: v for k, v in evaluate_all.WEIGHTS.items()} | {"ngram": tmp_path / "lm.pkl", "bert_int8": tmp_path / "bert_int8.pt"}
    weights = {k: (REPO_ROOT / v if k not in ("ngram", "bert_int8") else v) for k, v in weights.items()}
    _tiny_lm(items).save(weights["ngram"])
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"params": _PARAMS}), encoding="utf-8")
    outs = {"json": tmp_path / "b.json", "md": tmp_path / "b.md", "failures": tmp_path / "f.json"}
    for name, value in (("WEIGHTS", weights), ("OUT", outs), ("PARAMS_PATH", params), ("WARMUP_ITEMS", 1), ("LATENCY_RUNS", 2), ("BertMLMReranker", _tiny_reranker)):
        monkeypatch.setattr(evaluate_all, name, value)
    monkeypatch.setattr(evaluate_all, "DEV_WORKSHEETS_PATH", worksheets)
    assert evaluate_all.main(["--worksheets", str(worksheets)]) == 0
    summary = json.loads(outs["json"].read_text(encoding="utf-8"))
    assert summary["ladder_n"] == 3 and set(summary["ladder"]) == set(evaluate_all._LABELS)
    assert {c["result"] for c in summary["checks"]} <= {"PASS", "FAIL"} and len(summary["checks"]) == 4
    assert summary["vision_n"] == 18800 and 85.0 < summary["vision"]["char_cnn_fp32"]["acc_47_pct"] < 95.0
    assert "## Table 2" in outs["md"].read_text(encoding="utf-8")
    assert isinstance(json.loads(outs["failures"].read_text(encoding="utf-8")), list)
    assert weights["bert_int8"].is_file()
    assert "## Table 1" in capsys.readouterr().out


def test_evaluate_rejects_empty_worksheet_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    assert evaluate_all.main(["--worksheets", str(empty), "--no-save"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "non-empty" in err


def test_evaluate_main_reports_value_errors_in_one_line(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def bad(*_a: Any, **_k: Any) -> Any:
        raise KeyError("params")

    monkeypatch.setattr(evaluate_all, "evaluate", bad)
    assert evaluate_all.main([]) == 1
    assert capsys.readouterr().err.count("\n") == 1


def test_tune_bert_run_writes_params_from_dev_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    items = [it for it in _dev_items(50) if it["target_word"].isalpha()][:4]  # numeric items are decided by the guard, not the weights
    lm = _tiny_lm(items)
    monkeypatch.setattr(train_ngram, "_DEV_PATH", DEV_PATH)
    monkeypatch.setattr(train_ngram, "compute_dev_cnn_probs", lambda dev, **_k: [_onehot_probs(it["target_word"]) for it in dev])
    def dev_only(path: Path) -> list[dict[str, Any]]:
        assert Path(path) == DEV_PATH, f"tuning read {path}"
        return items

    monkeypatch.setattr(train_ngram, "_load_worksheet_items", dev_only)
    monkeypatch.setattr(train_ngram, "_TEST_PATH", Path("must-not-be-read.json"))
    monkeypatch.setattr(tune_bert.ClassicalNLPCorrector, "load", classmethod(lambda cls, *a, **k: lm))
    monkeypatch.setattr(tune_bert, "BertMLMReranker", _tiny_reranker)
    monkeypatch.setattr(tune_bert, "BERT_ALPHAS", (0.0, 1.0))
    monkeypatch.setattr(tune_bert, "SIMPLEX_STEP", 0.25)
    out = tmp_path / "params.json"
    assert tune_bert.main(["--out", str(out)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["dev_items"] == 4 and set(saved["params"]) == {"alpha", "lambda_vis", "lambda_bert", "gamma"}
    assert {"objective", "tie_break", "scale", "params_on_grid_edge", "int8_alone_params", "int8_alone_would_differ", "objective_value_int8_with_fp32_params"} <= set(saved)
    assert sum(saved["params"][k] for k in ("lambda_vis", "lambda_bert", "gamma")) == pytest.approx(1.0)
    assert "BERT fusion tuned" in capsys.readouterr().out


def test_tune_bert_main_reports_value_errors_in_one_line(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def bad(*_a: Any, **_k: Any) -> Any:
        raise ValueError("no dev items")

    monkeypatch.setattr(tune_bert, "run", bad)
    assert tune_bert.main([]) == 1
    assert capsys.readouterr().err == "Error: no dev items\n"

"""End-to-end tests of the benchmark and BERT-tuning scripts with tiny models (no network, no test-set pass)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

from data.preprocess import CANONICAL_36_CHARS
from models.bert_reranker import BertMLMReranker
from models.classical_nlp import ClassicalNLPCorrector, tokenize
from scripts import evaluate_all, train_ngram
from scripts import tune_bert_fusion as tune_bert

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


def _setup_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, params_doc: dict[str, Any] | None = None) -> tuple[Path, dict[str, Path], dict[str, Path]]:
    items = _dev_items(3)
    worksheets = tmp_path / "ws.json"
    worksheets.write_text(json.dumps(items), encoding="utf-8")
    weights = {k: (REPO_ROOT / v if k not in ("ngram", "bert_int8") else v) for k, v in evaluate_all.WEIGHTS.items()}
    weights |= {"ngram": tmp_path / "lm.pkl", "bert_int8": tmp_path / "bert_int8.pt"}
    _tiny_lm(items).save(weights["ngram"])
    params = tmp_path / "params.json"
    params.write_text(json.dumps(params_doc if params_doc is not None else {"params": _PARAMS, "tuned_on": evaluate_all.EXPECTED_TUNING_FILE}), encoding="utf-8")
    outs = {"json": tmp_path / "b.json", "md": tmp_path / "b.md", "failures": tmp_path / "f.json", "predictions": tmp_path / "p.json"}
    for name, value in (("WEIGHTS", weights), ("OUT", outs), ("PARAMS_PATH", params), ("WARMUP_ITEMS", 1), ("LATENCY_RUNS", 2), ("BertMLMReranker", _tiny_reranker)):
        monkeypatch.setattr(evaluate_all, name, value)
    monkeypatch.setattr(evaluate_all, "DEV_WORKSHEETS_PATH", worksheets)
    return worksheets, weights, outs


@needs_assets
def test_evaluate_writes_all_outputs_with_tiny_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    worksheets, weights, outs = _setup_eval(tmp_path, monkeypatch)
    assert evaluate_all.main(["--worksheets", str(worksheets)]) == 0
    summary = json.loads(outs["json"].read_text(encoding="utf-8"))
    assert summary["ladder_n"] == 3 and set(summary["ladder"]) == set(evaluate_all._LABELS)
    assert {c["result"] for c in summary["checks"]} <= {"PASS", "FAIL"} and len(summary["checks"]) == 5
    assert summary["vision_n"] == 18800 and 85.0 < summary["vision"]["char_cnn_fp32"]["acc_47_pct"] < 95.0
    assert "## Table 2" in outs["md"].read_text(encoding="utf-8")
    assert isinstance(json.loads(outs["failures"].read_text(encoding="utf-8")), list)
    assert weights["bert_int8"].is_file()
    from scripts import recompute_metrics

    assert recompute_metrics.main(["--predictions", str(outs["predictions"]), "--results", str(outs["json"])]) == 0  # every reported number re-derives
    assert "## Table 1" in capsys.readouterr().out


@needs_assets
def test_no_save_writes_no_files_and_int8_size_comes_from_the_loaded_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from models.quantization import get_model_disk_size_mb

    worksheets, weights, outs = _setup_eval(tmp_path, monkeypatch)
    assert evaluate_all.main(["--worksheets", str(worksheets), "--no-save"]) == 0
    assert not weights["bert_int8"].exists() and not any(p.exists() for p in outs.values())
    capsys.readouterr()
    weights["bert_int8"].write_bytes(b"old")  # a stale cached file must not be what gets reported
    fp32 = _tiny_reranker()
    models = {"bert_fp32": fp32, "bert_int8": fp32.to_int8()}
    sizes = evaluate_all.row_sizes(models)
    assert sizes["bert_int8_only"] == pytest.approx(get_model_disk_size_mb(models["bert_int8"].model))


@pytest.mark.parametrize(("doc", "fragment"), [({"tuned_on": "data/worksheets_dev_50.json"}, "params"),
                                               ({"params": {"alpha": 1.0}, "tuned_on": "data/worksheets_dev_50.json"}, "params"),
                                               ({"params": _PARAMS, "tuned_on": "data/worksheets_test_300.json"}, "tuned on")])
def test_evaluate_rejects_bad_fusion_params_file(doc: dict[str, Any], fragment: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    worksheets, _, _ = _setup_eval(tmp_path, monkeypatch, params_doc=doc)
    assert evaluate_all.main(["--worksheets", str(worksheets), "--no-save"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and fragment in err and "Error: 'params'" not in err


def _write_items(path: Path, split: str) -> Path:
    items = [{**it, "emnist_split": split} for it in _dev_items(2)]
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


@pytest.mark.parametrize("split", ["train", "val"])
def test_test_file_must_use_emnist_test_images(split: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = _write_items(tmp_path / "worksheets_test_300.json", split)
    monkeypatch.setattr(evaluate_all, "TEST_WORKSHEETS_PATH", test_file)
    monkeypatch.setattr(evaluate_all, "load_models", lambda *_a, **_k: pytest.fail("models must not load before validation"))
    assert evaluate_all.main(["--worksheets", str(test_file), "--no-save"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "emnist_split" in err and "'test'" in err


def test_unknown_split_and_bad_warmup_file_fail_before_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(evaluate_all, "load_models", lambda *_a, **_k: pytest.fail("models must not load before validation"))
    assert evaluate_all.main(["--worksheets", str(_write_items(tmp_path / "custom.json", "train")), "--no-save"]) == 1
    assert "emnist_split" in capsys.readouterr().err
    monkeypatch.setattr(evaluate_all, "DEV_WORKSHEETS_PATH", _write_items(tmp_path / "dev.json", "test"))
    assert evaluate_all.main(["--worksheets", str(_write_items(tmp_path / "ok.json", "test")), "--no-save"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "'val'" in err


def test_evaluate_rejects_items_without_id_or_split(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "ws.json"
    bad.write_text(json.dumps([{"sentence": "The [MASK] is hot.", "target_word": "SUN"}]), encoding="utf-8")
    assert evaluate_all.main(["--worksheets", str(bad), "--no-save"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "emnist_split" in err


@pytest.mark.parametrize(("module", "target"), [(evaluate_all, "evaluate"), (tune_bert, "run")])
def test_model_download_errors_print_one_line(module: Any, target: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def offline(*_a: Any, **_k: Any) -> Any:
        raise OSError("We couldn't connect to 'https://huggingface.co' to load the files.\nCheck your internet connection.")

    monkeypatch.setattr(module, target, offline)
    assert module.main([]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "File system error" not in err and "download" in err


def test_errno_less_os_errors_are_not_called_downloads_unless_from_hugging_face() -> None:
    from data._io_errors import describe_cli_os_error

    assert describe_cli_os_error(OSError("handle\nclosed")) == "Error: handle closed"
    hf = describe_cli_os_error(OSError("We couldn't connect to 'https://huggingface.co'.\nRetry."))
    assert hf.startswith("Model download or cache error") and "\n" not in hf


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
    monkeypatch.setattr(tune_bert, "DEV_WORKSHEETS_PATH", DEV_PATH)
    monkeypatch.setattr(train_ngram, "compute_dev_cnn_probs", lambda dev, **_k: [_onehot_probs(it["target_word"]) for it in dev])
    def dev_only(path: Path) -> list[dict[str, Any]]:
        assert Path(path) == DEV_PATH, f"tuning read {path}"
        return items

    monkeypatch.setattr(tune_bert, "load_worksheet_items", dev_only)
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

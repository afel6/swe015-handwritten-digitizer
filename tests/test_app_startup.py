"""`python app.py` startup failures: each prints exactly one stderr line naming the problem and the fix, exit 1, no traceback."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = REPO_ROOT / "weights"
REQUIRED = ("dense_mlp.pt", "char_cnn_fp32.pt", "char_cnn_int8.pt", "ngram_lm.pkl")
needs_weights = pytest.mark.skipif(not all((WEIGHTS / n).is_file() for n in REQUIRED), reason="model weights not downloaded")
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub" / "models--bert-base-uncased"
needs_hf_cache = pytest.mark.skipif(not HF_CACHE.is_dir(), reason="bert-base-uncased not in the local Hugging Face cache")


def _start(weights_dir: Path, **env: str) -> subprocess.CompletedProcess[str]:
    full_env = {**os.environ, "SWE015_WEIGHTS_DIR": str(weights_dir), **env}
    return subprocess.run([sys.executable, "-W", "error", str(REPO_ROOT / "app.py")], cwd=REPO_ROOT, env=full_env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, check=False)


def _one_line(proc: subprocess.CompletedProcess[str]) -> str:
    assert proc.returncode == 1, proc.stderr
    lines = proc.stderr.strip().splitlines()
    assert len(lines) == 1 and "Traceback" not in proc.stderr, proc.stderr
    return lines[0]


def _copy_weights(tmp_path: Path) -> Path:
    target = tmp_path / "weights"
    target.mkdir()
    for name in REQUIRED:
        shutil.copyfile(WEIGHTS / name, target / name)
    return target


def test_missing_weights_exit_with_one_line_and_fix_command(tmp_path: Path) -> None:
    line = _one_line(_start(tmp_path))
    assert "dense_mlp.pt" in line and "python scripts/download_assets.py" in line


def test_a_directory_where_a_weight_file_belongs_is_reported(tmp_path: Path) -> None:
    (tmp_path / "dense_mlp.pt").mkdir()
    line = _one_line(_start(tmp_path))
    assert "dense_mlp.pt" in line and "python scripts/download_assets.py" in line


@needs_weights
def test_a_corrupt_weight_file_is_reported(tmp_path: Path) -> None:
    weights = _copy_weights(tmp_path)
    (weights / "char_cnn_fp32.pt").write_bytes(b"not a checkpoint")
    line = _one_line(_start(weights))
    assert "char_cnn_fp32.pt" in line and "python scripts/download_assets.py" in line


@needs_weights
def test_offline_without_the_bert_model_is_reported(tmp_path: Path) -> None:
    weights = _copy_weights(tmp_path)
    line = _one_line(_start(weights, HF_HOME=str(tmp_path / "empty_hf"), HF_HUB_OFFLINE="1"))
    assert "bert-base-uncased" in line


@needs_weights
@needs_hf_cache
def test_an_unwritable_int8_cache_is_reported(tmp_path: Path) -> None:
    weights = _copy_weights(tmp_path)
    (weights / "bert_int8_dynamic.pt").mkdir()
    line = _one_line(_start(weights, HF_HUB_OFFLINE="1"))
    assert "bert_int8_dynamic.pt" in line

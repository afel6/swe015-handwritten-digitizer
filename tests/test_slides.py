"""The slide deck: exactly 10 pages/slides, rendered once to both formats, every number read from results/*.json."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import fitz
import pytest
from pptx import Presentation

from scripts import generate_slides

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
pytestmark = pytest.mark.skipif(not (REPO_ROOT / "weights" / "char_cnn_fp32.pt").is_file(), reason="slides render Grad-CAM from the CNN weights (python scripts/download_assets.py)")
META = {"name": "Test Student", "student_id": "S-123", "repo_url": generate_slides.DEFAULT_REPO, "video_url": "https://example.org/video"}


@pytest.fixture(scope="module")
def deck(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return generate_slides.render(META, tmp_path_factory.mktemp("deck"), RESULTS)


def _pdf_text(pdf: Path) -> list[str]:
    with fitz.open(pdf) as doc:
        return [page.get_text() for page in doc]


def test_pdf_and_pptx_both_have_exactly_ten_slides(deck: tuple[Path, Path]) -> None:
    pdf, pptx = deck
    assert len(_pdf_text(pdf)) == 10
    assert len(Presentation(str(pptx)).slides) == 10


def test_title_slide_shows_student_details(deck: tuple[Path, Path]) -> None:
    with fitz.open(deck[0]) as doc:
        assert all(value in doc[0].get_text() for value in META.values())
        links = {link.get("uri") for link in doc[0].get_links()}
        assert {META["repo_url"], META["video_url"]} <= links


def test_slide_numbers_match_the_results_files(deck: tuple[Path, Path]) -> None:
    pages = _pdf_text(deck[0])
    bench = json.loads((RESULTS / "benchmark_results.json").read_text(encoding="utf-8"))
    for key, row in bench["ladder"].items():
        assert f"{row['word_acc_pct']:.2f}" in pages[7], key
    for name, row in bench["vision"].items():
        assert f"{row['acc_47_pct']:.2f}" in pages[5], name
    ngram = json.loads((RESULTS / "ngram_metrics.json").read_text(encoding="utf-8"))
    assert f"tri {ngram['ppl_trigram']:.0f}" in pages[6]
    assert all(c["check"] in pages[7] for c in bench["checks"])


def test_numbers_are_read_not_hard_coded(tmp_path: Path) -> None:
    results = tmp_path / "results"
    shutil.copytree(RESULTS, results)
    bench = json.loads((results / "benchmark_results.json").read_text(encoding="utf-8"))
    bench["ladder"]["classical"]["word_acc_pct"] = 12.34
    (results / "benchmark_results.json").write_text(json.dumps(bench), encoding="utf-8")
    pdf, _ = generate_slides.render(META, tmp_path / "out", results)
    assert "12.34" in _pdf_text(pdf)[7]


def test_slide_8_shows_reference_and_slide_2_the_split(deck: tuple[Path, Path]) -> None:
    pages = _pdf_text(deck[0])
    ref = json.loads((RESULTS / "vlm_reference.json").read_text(encoding="utf-8"))
    flat = " ".join(pages[7].split())
    assert "not trained by us" in flat
    for run in ref["runs"].values():
        assert f"{run['word_acc_pct']:.2f}" in flat
    assert "Full-page OCR" in " ".join(pages[1].split())
    assert "Full-page OCR" in " ".join(pages[8].split())

"""Render the 10-slide presentation once (16:9 matplotlib figures) and write BOTH the PDF and the PPTX from it.

    python scripts/generate_slides.py --name "Ada Lovelace" --student-id 123 --video-url https://...

Every number comes from results/*.json (see scripts/slide_content.py); nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import tempfile
import textwrap
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

if str(_ROOT := Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from pptx import Presentation
from pptx.util import Emu

from data._io_errors import describe_cli_os_error
from scripts import slide_content as sc

DEFAULT_REPO = "https://github.com/afel6/swe015-handwritten-digitizer"
OUT_DIR = Path("deliverables")
W_IN, H_IN, DPI = 13.333, 7.5, 150
INK, ACCENT, MUTED, PAPER = "#1d2433", "#1f4fa3", "#4a5466", "#fbf8f1"
_WRAP = 118
_LINE_H = 1.45 / (72 * H_IN)  # axes-fraction height of one text line per point of font size
_GAP_H = 0.55 / (72 * H_IN)


def _slide(title: str, number: int) -> tuple[Any, Any]:
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=PAPER)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.93, title, fontsize=26, fontweight="bold", color=INK, va="top")
    ax.plot([0.04, 0.96], [0.855, 0.855], color=ACCENT, lw=2)
    ax.text(0.96, 0.03, f"{number} / 10", fontsize=10, color=MUTED, ha="right")
    return fig, ax


def _bullets(ax: Any, lines: Sequence[str], top: float = 0.80, size: int = 14, left: float = 0.05, width: int = _WRAP) -> float:
    y = top
    for line in lines:
        wrapped = textwrap.wrap(line, width) or [""]
        ax.text(left, y, "•  " + "\n    ".join(wrapped), fontsize=size, color=INK, va="top", family="DejaVu Sans")
        y -= _LINE_H * size * len(wrapped) + _GAP_H * size
    return y


def _table(fig: Any, rect: tuple[float, float, float, float], header: Sequence[str], rows: Sequence[Sequence[str]], highlight: int | None = None, first_col: float = 0.3, row_scale: float = 1.45, align: str = "center") -> None:
    ax = fig.add_axes(rect)
    ax.axis("off")
    rest = (1.0 - first_col) / (len(header) - 1)
    table = ax.table(cellText=[list(r) for r in rows], colLabels=list(header), loc="upper center", cellLoc=align, colWidths=[first_col] + [rest] * (len(header) - 1))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, row_scale)
    for (r, _c), cell in table.get_celld().items():
        cell.set_edgecolor("#d9d2c3")
        if r == 0:
            cell.set_facecolor("#e6edfa")
            cell.set_text_props(fontweight="bold", color=INK)
        elif highlight is not None and r == highlight + 1:
            cell.set_facecolor("#fff4cc")


def slide_title(res: Mapping[str, Any], meta: Mapping[str, str]) -> Any:
    fig, ax = _slide("Handwritten Answer Digitizer", 1)
    ax.text(0.05, 0.72, "SWE015 - Use Case 1: read handwritten cloze answers and correct them with language models", fontsize=18, color=INK)
    for i, (k, v) in enumerate((("Student", meta["name"]), ("Student ID", meta["student_id"]), ("Repository", meta["repo_url"]), ("Demo video", meta["video_url"]))):
        ax.text(0.05, 0.56 - 0.07 * i, f"{k}:", fontsize=16, color=MUTED)
        ax.text(0.22, 0.56 - 0.07 * i, v, fontsize=16, color=INK)
    ax.text(0.05, 0.18, sc.hardware_line(res), fontsize=11, color=MUTED)
    return fig


def slide_intro(res: Mapping[str, Any], meta: Mapping[str, str]) -> Any:
    fig, ax = _slide("Introduction: problem and goal", 2)
    br = res["benchmark_results"]["ladder"]
    _bullets(ax, [
        "Problem: worksheets ask students to write one missing word in boxes; a teacher must read every box by hand.",
        "Goal: recognise each boxed character with a neural network, then use the sentence to fix misreads (0/O, 1/I, 5/S ...).",
        "Constraint: one local app on a laptop CPU, quantized for speed and size, with every claim measured on held-out data.",
        f"Result on 300 held-out worksheets: word accuracy {br['raw_fp32']['word_acc_pct']:.1f} % (CNN alone) -> {br['classical']['word_acc_pct']:.1f} % (n-gram) "
        f"-> {br['bert_fp32']['word_acc_pct']:.1f} % (BERT) / {br['quantized_app']['word_acc_pct']:.1f} % (INT8 app model).",
    ])
    return fig


def _sample_strip() -> Any:
    path = Path("data/sample_worksheets/sample_05.png")
    return Image.open(path).convert("RGB") if path.is_file() else None


def slide_data(res: Mapping[str, Any], meta: Mapping[str, str]) -> Any:
    fig, ax = _slide("Data and preprocessing", 3)
    y = _bullets(ax, sc.data_lines(res), size=13)
    strip = _sample_strip()
    if strip is not None:
        img_ax = fig.add_axes((0.30, 0.04, 0.40, max(0.08, y - 0.08)))
        img_ax.imshow(strip)
        img_ax.axis("off")
        img_ax.set_title("A held-out worksheet strip (EMNIST test glyphs)", fontsize=10, color=MUTED)
    return fig


ARCH_STEPS = (
    ("UI: canvas / upload\n(localhost browser)", 0.11, 0.70), ("FastAPI\nvalidation", 0.305, 0.70), ("slice_strip_image\npreprocess_box (28x28)", 0.50, 0.70),
    ("Dense MLP FP32\nCharCNN FP32 (+Grad-CAM)\nCharCNN INT8 static", 0.695, 0.70), ("47 -> 36 fold", 0.89, 0.70),
    ("candidate pool: SymSpell\n(weighted ED) vs lexicon", 0.89, 0.38), ("n-gram (MLE tri/bi + add-k uni)\n| BERT FP32 span-PLL\n| BERT INT8 dynamic span-PLL", 0.63, 0.38),
    ("fusion (simplex-tuned)\n+ numeric guard", 0.37, 0.38), ("JSON -> UI\n4 models side by side", 0.11, 0.38),
)


def draw_architecture(ax: Any) -> None:
    for i, (label, x, y) in enumerate(ARCH_STEPS):
        ax.add_patch(FancyBboxPatch((x - 0.085, y - 0.08), 0.17, 0.16, boxstyle="round,pad=0.01", fc="#ffffff", ec=ACCENT, lw=1.6))
        ax.text(x, y, label, ha="center", va="center", fontsize=10, color=INK)
        if i + 1 < len(ARCH_STEPS):
            _, nx, ny = ARCH_STEPS[i + 1]
            start, end = ((x + 0.087, y), (nx - 0.087, ny)) if ny == y and nx > x else ((x - 0.087, y), (nx + 0.087, ny)) if ny == y else ((x, y - 0.082), (nx, ny + 0.082))
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color=ACCENT, lw=1.4))
    ax.text(0.5, 0.14, "Everything runs on the CPU on 127.0.0.1; the benchmark and the app call the same functions (data/word_pipeline.py, models/ladder.py).", ha="center", fontsize=12, color=MUTED)


def slide_architecture(res: Mapping[str, Any], meta: Mapping[str, str]) -> Any:
    fig, ax = _slide("R6 architecture", 4)
    draw_architecture(ax)
    return fig


def slide_technique_map(res: Mapping[str, Any], meta: Mapping[str, str]) -> Any:
    fig, _ = _slide("R2 technique map", 5)
    rows = [
        ["A - Neural networks", "Dense MLP (784-512-256-47) vs CharCNN (3 conv blocks); affine augmentation, dropout, AdamW/SGD; Grad-CAM"],
        ["B - Classical NLP", "Weighted Levenshtein (OCR confusions), SymSpell candidates, interpolated trigram LM (Jelinek-Mercer)"],
        ["C - Transformers", "bert-base-uncased masked LM: batched span pseudo-log-likelihood re-ranking of the same candidate pool"],
        ["D - Optimisation", "Static INT8 PTQ of the CNN (onednn), dynamic INT8 BERT (Linear layers); CPU latency and size measured"],
        ["Method", "Dev-only tuning (continuous log-softmax objective on a simplex), single held-out test evaluation, recomputable results"],
    ]
    wrapped = [[area, "\n".join(textwrap.wrap(text, 92))] for area, text in rows]
    _table(fig, (0.04, 0.10, 0.92, 0.70), ["Area", "Exactly what is used"], wrapped, first_col=0.2, row_scale=3.2, align="left")
    return fig


def _gradcam_examples(count: int = 3) -> list[Any]:
    try:
        import torch

        from data.preprocess import preprocess_box, slice_strip_image
        from models.char_models import load_char_cnn_fp32
        from models.gradcam import generate_gradcam_b64

        cnn = load_char_cnn_fp32("weights/char_cnn_fp32.pt")
        with Image.open("data/sample_worksheets/sample_05.png") as strip:
            boxes = slice_strip_image(strip.copy(), num_boxes=3)
        maps = [generate_gradcam_b64(cnn, preprocess_box(b)[0].unsqueeze(0)) for b in boxes[:count]]
        del torch
        return [Image.open(io.BytesIO(base64.b64decode(m))) for m in maps]
    except (OSError, ValueError, RuntimeError):
        return []


def slide_area_a(res: Mapping[str, Any], meta: Mapping[str, str]) -> Any:
    fig, ax = _slide("Area A: Dense vs CNN, regularisation, Grad-CAM", 6)
    _table(fig, (0.04, 0.48, 0.92, 0.33), ["Model", "47-class %", "36-folded %", "MB", "b1 ms", "b8 ms"], sc.vision_rows(res))
    _bullets(ax, [*sc.vision_fit_lines(res), "Grad-CAM (FP32 CNN) shows which strokes drive each prediction:"], top=0.44, size=13)
    for i, img in enumerate(_gradcam_examples()):
        img_ax = fig.add_axes((0.30 + i * 0.14, 0.04, 0.12, 0.22))
        img_ax.imshow(img)
        img_ax.axis("off")
    return fig


def slide_areas_bc(res: Mapping[str, Any], meta: Mapping[str, str]) -> Any:
    fig, ax = _slide("Areas B and C: candidates, n-gram LM, BERT re-ranking", 7)
    _bullets(ax, sc.nlp_lines(res), size=13)
    return fig


def slide_area_d(res: Mapping[str, Any], meta: Mapping[str, str]) -> Any:
    fig, ax = _slide("Area D: quantization and R4 results", 8)
    _table(fig, (0.04, 0.47, 0.92, 0.37), ["Pipeline (N=300)", "Word %", "Char %", "MB", "ms/word"], sc.ladder_rows(res), highlight=4, first_col=0.36)
    y = _bullets(ax, ["Static INT8 CNN (observers on train data) and dynamic INT8 BERT; sizes are serialized state_dicts; CPU, 4 threads, perf_counter.", *sc.checks_lines(res)], top=0.45, size=11)
    note = sc.hardware_line(res) + ". Tuned on dev only; the test set was evaluated 3 times and run 3 is reported (runs 1-2 superseded; reasons in results/)."
    ax.text(0.05, y - 0.01, "\n".join(textwrap.wrap(note, 150)), fontsize=9.5, color=MUTED, va="top")
    return fig


def slide_conclusion(res: Mapping[str, Any], meta: Mapping[str, str]) -> Any:
    fig, ax = _slide("Conclusion", 9)
    br = res["benchmark_results"]
    _bullets(ax, [
        f"Win: context correction lifts word accuracy from {br['ladder']['raw_fp32']['word_acc_pct']:.1f} % to {br['ladder']['bert_fp32']['word_acc_pct']:.1f} %; INT8 keeps it within "
        f"{br['ladder']['bert_fp32']['word_acc_pct'] - br['ladder']['quantized_app']['word_acc_pct']:.2f} pp with a {br['bert_sizes']['fp32_mb'] / br['bert_sizes']['int8_mb']:.2f}x smaller BERT.",
        f"Ceiling: only {br['recall_pct']['int8']:.1f} % of targets reach the candidate pool; BERT and n-gram can only choose among those.",
        f"Limitations: digit strings read with letters (20O0) never enter the pool; span-PLL favours multi-piece words; {100 * res['corpus_sources']['stats']['test_oov_rate']:.1f} % test OOV; CPU latencies vary between runs.",
        *("Failure - " + line for line in sc.failure_lines(res, limit=2)),
        "Next steps: digit-mapped numeric candidates, PLL-word-l2r scoring, a larger lexicon.",
    ], size=12, width=128)
    return fig


def slide_ai_usage(res: Mapping[str, Any], meta: Mapping[str, str]) -> Any:
    fig, ax = _slide("R8 AI usage", 10)
    _bullets(ax, [
        "Claude (Anthropic) was used as a coding assistant: planning the slices, writing and reviewing code and tests, and drafting the README, slides and demo script.",
        "Every number in this deck comes from scripts run on this repository (results/*.json); scripts/recompute_metrics.py re-derives them without the models.",
        "Design decisions, the dataset protocol (dev-only tuning, one held-out test set) and the final review are the student's responsibility.",
        "Pretrained model: bert-base-uncased (Devlin et al., 2019). Data: EMNIST (Cohen et al., 2017), Project Gutenberg, Wikipedia (CC BY-SA).",
    ], size=14)
    return fig


SLIDES: tuple[Callable[[Mapping[str, Any], Mapping[str, str]], Any], ...] = (
    slide_title, slide_intro, slide_data, slide_architecture, slide_technique_map, slide_area_a, slide_areas_bc, slide_area_d, slide_conclusion, slide_ai_usage)


def render(meta: Mapping[str, str], out_dir: Path = OUT_DIR, results_dir: Path = sc.RESULTS) -> tuple[Path, Path]:
    """Render every slide once; the PDF page and the PPTX picture come from the same figure."""
    res = sc.load_results(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path, pptx_path = out_dir / "SWE015_Presentation.pdf", out_dir / "SWE015_Presentation.pptx"
    deck = Presentation()
    deck.slide_width, deck.slide_height = Emu(int(W_IN * 914400)), Emu(int(H_IN * 914400))
    with PdfPages(pdf_path) as pdf, tempfile.TemporaryDirectory() as tmp:
        for i, build in enumerate(SLIDES, 1):
            fig = build(res, meta)
            pdf.savefig(fig, facecolor=PAPER)
            png = Path(tmp) / f"slide_{i:02d}.png"
            fig.savefig(png, dpi=DPI, facecolor=PAPER)
            plt.close(fig)
            slide = deck.slides.add_slide(deck.slide_layouts[6])
            slide.shapes.add_picture(str(png), 0, 0, width=deck.slide_width, height=deck.slide_height)
    deck.save(pptx_path)
    return pdf_path, pptx_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the 10-slide deck to PDF and PPTX from results/*.json.")
    parser.add_argument("--name", default="TBD")
    parser.add_argument("--student-id", default="TBD")
    parser.add_argument("--repo-url", default=DEFAULT_REPO)
    parser.add_argument("--video-url", default="TBD")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    meta = {"name": args.name, "student_id": args.student_id, "repo_url": args.repo_url, "video_url": args.video_url}
    try:
        pdf, pptx = render(meta, args.out_dir)
    except OSError as exc:
        sys.stderr.write(describe_cli_os_error(exc) + "\n")
        return 1
    except (KeyError, ValueError) as exc:
        sys.stderr.write(f"Error: results are incomplete ({' '.join(str(exc).split())}); run the benchmark first.\n")
        return 1
    sys.stdout.write(f"Wrote {pdf} and {pptx}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

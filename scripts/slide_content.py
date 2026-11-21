"""Slide text and tables built from results/*.json (no metric is hard-coded here)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RESULTS = Path("results")
ROW_ORDER = ("raw_fp32", "raw_int8", "classical", "bert_fp32", "quantized_app", "ablation_fp32cnn_int8bert")
SHORT = {"raw_fp32": "(1) FP32 CNN, no correction", "raw_int8": "(1b) INT8 CNN, no correction", "classical": "(2) FP32 CNN + n-gram",
         "bert_fp32": "(3) FP32 CNN + FP32 BERT", "quantized_app": "(4) INT8 CNN + INT8 BERT [app]", "ablation_fp32cnn_int8bert": "ablation: FP32 CNN + INT8 BERT"}


def load_results(root: Path = RESULTS) -> dict[str, Any]:
    names = ("benchmark_results", "ngram_metrics", "bert_fusion_params", "corpus_sources", "vision_metrics", "failure_examples",
             "benchmark_run1_superseded", "benchmark_run2_superseded")
    out = {n: json.loads((root / f"{n}.json").read_text(encoding="utf-8")) for n in names}
    if (root / "vlm_reference.json").is_file():  # optional outside-model reference
        out["vlm_reference"] = json.loads((root / "vlm_reference.json").read_text(encoding="utf-8"))
    return out


def reference_line(res: Mapping[str, Any]) -> str:
    """The Qwen3-VL reference in one line, read from results/vlm_reference.json; empty when it is absent."""
    ref = res.get("vlm_reference")
    if not ref:
        return ""
    strip, sent = ref["runs"]["strip_only"], ref["runs"]["strip_sentence"]
    ms = (strip["latency_mean_ms"] + sent["latency_mean_ms"]) / 2
    return (f"Reference, not trained by us: Qwen3-VL 8B (6.2 GB, {ref['gpu']}), zero-shot on the same {ref['n']} boxed strips: "
            f"{strip['word_acc_pct']:.2f} % word acc from the strip alone, {sent['word_acc_pct']:.2f} % with the sentence, about {ms:.0f} ms/word on the GPU")


def vision_rows(res: Mapping[str, Any]) -> list[list[str]]:
    vis = res["benchmark_results"]["vision"]
    names = {"dense_mlp_fp32": "Dense MLP FP32", "char_cnn_fp32": "CharCNN FP32", "char_cnn_int8": "CharCNN INT8 (static)"}
    return [[names[k], f"{v['acc_47_pct']:.2f}", f"{v['acc_36_pct']:.2f}", f"{v['disk_mb']:.2f}", f"{v['latency_b1_ms']:.2f}", f"{v['latency_b8_ms']:.2f}"] for k, v in vis.items()]


def ladder_rows(res: Mapping[str, Any]) -> list[list[str]]:
    lad = res["benchmark_results"]["ladder"]
    return [[SHORT[k], f"{lad[k]['word_acc_pct']:.2f}", f"{lad[k]['char_acc_pct']:.2f}", f"{lad[k]['size_mb']:.1f}", f"{lad[k]['latency_mean_ms']:.1f}"] for k in ROW_ORDER]


def run_history_rows(res: Mapping[str, Any]) -> list[list[str]]:
    runs = [res["benchmark_run1_superseded"]["ladder"], res["benchmark_run2_superseded"]["ladder"], res["benchmark_results"]["ladder"]]
    return [[SHORT[k], *(f"{r[k]['word_acc_pct']:.2f}" for r in runs)] for k in ("classical", "bert_fp32", "quantized_app")]


def hardware_line(res: Mapping[str, Any]) -> str:
    hw = res["benchmark_results"]["hardware"]
    cuda = " (CUDA build, GPU unused for timing)" if "+cu" in str(hw["torch"]) else ""
    return f"CPU only: {hw['cpu_model']}, {hw['torch_threads']} torch threads, {hw['ram_gb']} GB RAM; torch {hw['torch']}{cuda}, transformers {hw['transformers']}"


def checks_lines(res: Mapping[str, Any]) -> list[str]:
    return [f"{c['result']}  {c['check']}: {c['value']}" for c in res["benchmark_results"]["checks"]]


def data_lines(res: Mapping[str, Any]) -> list[str]:
    split = res["vision_metrics"]["split_counts"]
    stats = res["corpus_sources"]["stats"]
    words = stats["word_counts"]
    return [
        f"EMNIST Balanced (47 classes): train {split['train']:,} / val {split['val']:,} (stratified 90/10, seed 42) / test {split['test']:,}",
        "47 -> 36 fold: lowercase letters merged into uppercase (case is not graded); probabilities summed",
        "Boxes: strip slicing -> border trim -> polarity -> tight crop -> 28x28 centring (same path in app, tuning and benchmark)",
        f"Language corpus: Gutenberg + Wikipedia, {words['train']:,} train / {words['dev']:,} dev / {words['ppl_test']:,} PPL-test words (disjoint sources)",
        f"Worksheets: {res['bert_fusion_params']['dev_items']} dev (EMNIST val glyphs) and {res['benchmark_results']['ladder_n']} held-out test strips (EMNIST test glyphs); "
        f"lexicon {stats['lexicon_size']:,} words",
        f"Test OOV rate (target not in lexicon): {100 * stats['test_oov_rate']:.1f} %",
    ]


def nlp_lines(res: Mapping[str, Any]) -> list[str]:
    ng, bp, br = res["ngram_metrics"], res["bert_fusion_params"], res["benchmark_results"]
    w, p = ng["fusion_weights"], bp["params"]
    return [
        "Candidates: CNN top-2 lattice -> SymSpell (edit distance <= 2) over the lexicon -> 20-word pool (fixed rule: vis - 2.0 x weighted ED)",
        f"Interpolated trigram LM (MLE tri/bi + add-k uni): held-out PPL uni {ng['ppl_unigram']:.0f} > bi {ng['ppl_bigram']:.0f} > tri {ng['ppl_trigram']:.0f}",
        f"n-gram fusion (dev-tuned, weights sum to 1): vis {w['vis_weight']:.2f}, edit {w['edit_penalty']:.2f}, LM {w['lm_weight']:.2f}",
        "BERT re-ranker: PLL_norm(w) = (1/k^alpha) sum_j log P(s_j | context, s_-j); 1 pass for 1-piece words, 1 batched pass for the rest",
        f"BERT fusion (dev-tuned): alpha {p['alpha']:.1f}, vis {p['lambda_vis']:.2f}, BERT {p['lambda_bert']:.2f}, edit {p['gamma']:.2f}",
        f"Candidate recall on test (ceiling for rows 2-4): FP32 pool {br['recall_pct']['fp32']:.2f} %, INT8 pool {br['recall_pct']['int8']:.2f} %",
    ]


def vision_fit_lines(res: Mapping[str, Any]) -> list[str]:
    models = res["vision_metrics"]["models"]
    cnn, mlp = models["char_cnn_fp32"], models["dense_mlp_fp32"]
    return [f"CNN beats the Dense MLP on EMNIST test; with random affine augmentation and dropout the CNN's best validation accuracy "
            f"({cnn['best_val_acc_47_pct']:.2f} %) stays close to its test accuracy ({cnn['test_acc_47_pct']:.2f} %); MLP {mlp['best_val_acc_47_pct']:.2f} / {mlp['test_acc_47_pct']:.2f} %."]


def failure_lines(res: Mapping[str, Any], limit: int = 3) -> list[str]:
    fails = res["failure_examples"]
    in_pool = [f for f in fails if f["target_in_pool_int8"]]
    numeric = [f for f in fails if f["target"].isdigit() and not f["target_in_pool_int8"]]
    picked = in_pool[:1] + numeric[:1] + [f for f in fails if f not in in_pool[:1] + numeric[:1]]
    return [f"{f['id']}: target {f['target']}, app read {f['raw_int8']} -> {f['outputs']['quantized_app']} ({f['why']})" for f in picked[:limit]]


def technique_rows(res: Mapping[str, Any]) -> list[list[str]]:
    """R2 map in the rubric's terms, each with the file that implements it."""
    ng = res["ngram_metrics"]
    return [
        ["A - Neural networks", "Feedforward net trained with SGD + backprop (models/char_models.py, scripts/train_vision.py); CNN with convolution + pooling; "
                                "data augmentation (random affine) + dropout; Grad-CAM explanations (models/gradcam.py)"],
        ["B - Classical NLP", "Regex tokenisation + text normalisation (models/classical_nlp.py, models/cloze.py); minimum weighted edit distance for spelling "
                              "candidates (SymSpell); n-gram LM with smoothing (add-k unigram + Jelinek-Mercer interpolation); perplexity "
                              f"{ng['ppl_unigram']:.1f} / {ng['ppl_bigram']:.1f} / {ng['ppl_trigram']:.1f} (uni / bi / tri)"],
        ["C - Transformers", "BERT masked-LM (attention-based Transformer) re-ranking with span pseudo-log-likelihood; WordPiece subword tokenizer "
                             "(models/bert_reranker.py)"],
        ["D - Optimisation", "Static INT8 post-training quantization of the CNN + dynamic INT8 BERT (models/quantization.py); benchmarking of size, latency "
                             "and accuracy (scripts/evaluate_all.py)"],
    ]


def conclusion_lines(res: Mapping[str, Any]) -> list[str]:
    """Wins and losses, every number read from the benchmark results."""
    lad, rec = res["benchmark_results"]["ladder"], res["benchmark_results"]["recall_pct"]
    raw, ng, b32, app = (lad[k] for k in ("raw_fp32", "classical", "bert_fp32", "quantized_app"))
    oov = 100 * res["corpus_sources"]["stats"]["test_oov_rate"]
    return [
        f"Win: context correction lifts word accuracy from {raw['word_acc_pct']:.2f} % (CNN alone) to {ng['word_acc_pct']:.2f} % (n-gram) and {b32['word_acc_pct']:.2f} % (BERT FP32).",
        f"Loss: FP32 BERT costs {b32['latency_mean_ms'] / ng['latency_mean_ms']:.1f}x the n-gram latency ({b32['latency_mean_ms']:.1f} vs {ng['latency_mean_ms']:.1f} ms/word) and "
        f"{b32['size_mb'] / ng['size_mb']:.1f}x the size ({b32['size_mb']:.1f} vs {ng['size_mb']:.1f} MiB) for {b32['word_acc_pct'] - ng['word_acc_pct']:+.2f} pp word accuracy on these strips.",
        f"Win: the INT8 app model is {b32['latency_mean_ms'] / app['latency_mean_ms']:.1f}x faster than FP32 BERT ({app['latency_mean_ms']:.1f} vs {b32['latency_mean_ms']:.1f} ms) and "
        f"about half the size ({app['size_mb']:.1f} vs {b32['size_mb']:.1f} MiB) at {app['word_acc_pct']:.2f} % word accuracy. "
        f"INT8 adds {app['word_acc_pct'] - ng['word_acc_pct']:.2f} pp over n-gram on this set.",
        f"Limits in the default word pipeline: {rec['int8']:.1f} % candidate recall, {oov:.1f} % test OOV, and a span-PLL preference for multi-piece words.",
    ]


def methodology_note(res: Mapping[str, Any]) -> str:
    """Tuning/test protocol sentence with every count read from the results."""
    runs = 1 + sum(1 for key in ("benchmark_run1_superseded", "benchmark_run2_superseded") if key in res)
    passes = len(res["benchmark_results"].get("retime", {}).get("passes", []))
    retime = f" Timing uses the median of {passes} passes by app latency." if passes else ""
    return (f"Tuned on dev only. Test run {runs} reported (see results/ for earlier runs). "
            f"Generated strips use known box counts. Latency covers the offline pipeline.{retime}")


def ai_usage_lines(name: str) -> list[str]:
    """R8 statement."""
    return [
        f"Built by {name} with assistance from Claude (Anthropic) and Codex (OpenAI).",
        "Student: chose the use case, made the design and evaluation decisions, ran all training and benchmarks on own laptops, reviewed the results.",
        "Claude: generated and reviewed code and tests, drafted the README and slides.",
        "Codex implemented review, digits-only recognition and the worksheet pilot tools. It helped prepare tests, documentation and the narrated demo.",
        "The original benchmark uses the student's runs. The metric checker recomputes statistics from the saved prediction records.",
        "Pretrained model: bert-base-uncased (Devlin et al., 2019). Data: EMNIST (Cohen et al., 2017), Project Gutenberg, Wikipedia (CC BY-SA).",
    ]

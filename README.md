# Handwritten Answer Digitizer

Author: Abdalraouf Alkabir (2309015482)

SWE015 Use Case 1. A student writes the missing word of a cloze sentence into boxes. The app reads every
box with a character CNN, then uses the sentence to correct misreads (0/O, 1/I, 5/S, ...) with an
interpolated n-gram model or a BERT re-ranker. It runs as one local web app on a laptop CPU, with an
INT8-quantized model as the default.

## Two parts

1. **Answer reader (the course project).** Everything above: our own trained CNNs, the n-gram model, BERT and
   quantization, measured on held-out worksheets. All graded numbers come from this part.
2. **Full-page OCR (the real-world tool).** A second tab in the same app reads whole handwritten or printed
   pages (English and Arabic) with Qwen3-VL 8B, a vision-language model **not trained by us**, running locally
   through Ollama, so documents never leave the laptop. It is optional: without Ollama the tab shows setup
   steps and the answer reader works as before. Setup: install Ollama from ollama.com/download, then
   `ollama pull qwen3-vl:8b-instruct-q4_K_M` (about 6 GB; a GPU is strongly recommended).

<!-- grader section: keep identical to AGENTS.md -->
## For graders

Everything below is a plain Python command; run them from the repository root in any shell. Supported Python: 3.12 and 3.13. A fresh virtual environment is recommended: `python -m venv .venv`, then activate it with PowerShell `.venv\Scripts\Activate.ps1`, cmd `.venv\Scripts\activate.bat`, or bash `source .venv/bin/activate`. CI runs the tests and `verify.py` on Ubuntu and Windows with both versions without the downloaded assets, and on Ubuntu and Windows with Python 3.13 after `python scripts/download_assets.py`.

1. Install (CPU build of torch first, then the pinned requirements):

       python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
       python -m pip install -r requirements.txt

2. Check every claim that needs no download (prints PASS/FAIL/SKIP per claim; under a minute without the weights, a few minutes once the weights are downloaded):

       python verify.py
       python -m pytest -q

   Without the model weights, tests that need them are skipped with a reason; in CI no test fails.

   If pytest or the app stops with `Windows fatal exception: code 0xc000001d`, set ONEDNN_MAX_CPU_ISA=AVX2 before starting Python: PowerShell `$env:ONEDNN_MAX_CPU_ISA="AVX2"`, cmd `set ONEDNN_MAX_CPU_ISA=AVX2`.

3. Optional, to run the weight-dependent checks and the app (downloads about 175 MB from this repository's public releases and checks every sha256; the app also downloads bert-base-uncased from Hugging Face on first start):

       python scripts/download_assets.py
       python verify.py
       python app.py

   The app serves http://127.0.0.1:8000.

| Claim | Result file | Command that recomputes or checks it |
|---|---|---|
| Every Table 1 and Table 2 number (accuracy, character accuracy, latency, candidate recall) | `results/benchmark_results.json`, `results/test_predictions.json` | `python scripts/recompute_metrics.py` |
| The vision models load and read the committed sample worksheet boxes | `weights/*.pt` (download) | `python verify.py` (check b) |
| The INT8 CNN checkpoint works on this platform's quantized engine and matches FP32 | `weights/char_cnn_int8.pt` (download) | `python verify.py` (check c) |
| Bad app requests get a one-line HTTP 400 without loading models | `app.py` | `python verify.py` (check d), `python -m pytest -q tests/test_app.py` |
| Held-out perplexity falls from unigram to bigram to trigram | `results/ngram_metrics.json` | `python -m pytest -q tests/test_nlp.py -k perplexity` |
| Fusion weights were tuned on the dev worksheets only | `results/bert_fusion_params.json` (`tuned_on`) | `python -m pytest -q tests/test_benchmark.py -k dev_only` |
| The test worksheets were evaluated three times; run 3 is reported | `results/benchmark_run1_superseded.md`, `results/benchmark_run2_superseded.md` | read the reason paragraph at the top of each file |
| Qwen3-VL reference (outside model, not part of the ladder) | `results/vlm_reference.json` | `python scripts/recompute_metrics.py` (re-running it needs Ollama and a GPU: `python scripts/benchmark_vlm.py`) |
<!-- end grader section -->

## What the app does

- **Input:** a sentence with one blank (`[MASK]` or `___`) and the answer as 1–16 hand-drawn boxes (canvas)
  or an uploaded PNG/JPEG strip. Fifteen held-out sample worksheets load with one click.
- **Per box:** the 28×28 image the models see, a Grad-CAM heatmap from the FP32 CNN, and top-2 guesses
  from the Dense MLP, the FP32 CNN and the INT8 CNN.
- **Per word:** four models side by side with their prediction, top candidates and latency: (1) the raw
  FP32 CNN read, (2) the n-gram corrector, (3) FP32 BERT, (4) the quantized app model (INT8 CNN + INT8
  BERT), which is the primary answer.

Setup: Python 3.12 or 3.13, a virtual environment, then the three install commands in the grader section
(`pip install` the CPU torch line, `pip install -r requirements.txt`, `python scripts/download_assets.py`).
Run with the single command `python app.py`. A GPU build of torch is optional: the app, the benchmark and
the tests run on the CPU.

## Architecture (R6)

```mermaid
flowchart LR
  UI["Browser UI<br/>canvas boxes or<br/>strip upload"]
  API["FastAPI on 127.0.0.1<br/>Host allow-list, JSON only,<br/>size / base64 / image checks"]
  SLICE["slice_strip_image<br/>(strip uploads)"]
  PRE["preprocess_box<br/>28x28 per box"]
  MLP["Dense MLP · FP32<br/>(display only)"]
  CNN32["CharCNN · FP32<br/>+ Grad-CAM"]
  CNN8["CharCNN · INT8<br/>static PTQ"]
  FOLD32["47 -> 36 fold<br/>FP32 lattice"]
  FOLD8["47 -> 36 fold<br/>INT8 lattice"]
  R1["Row 1: raw FP32 read"]
  POOL32["Candidate pool (FP32 read)<br/>SymSpell vs 22k lexicon,<br/>top 20 by vis - 2.0 x wED"]
  POOL8["Candidate pool (INT8 read)<br/>same fixed rule"]
  NG["Row 2: interpolated<br/>n-gram fusion"]
  B32["Row 3: BERT · FP32<br/>span-PLL fusion"]
  B8["Row 4: BERT · INT8 dynamic<br/>span-PLL fusion<br/>PRIMARY app output (R5)"]
  OUT["Numeric guard (rows 2-4)<br/>-> JSON -> UI:<br/>4-model table,<br/>Grad-CAM, top-2 badges"]
  UI --> API
  API -->|strip| SLICE
  API -->|boxes| PRE
  SLICE --> PRE
  PRE --> MLP
  PRE --> CNN32
  PRE --> CNN8
  CNN32 --> FOLD32
  CNN8 --> FOLD8
  FOLD32 -->|non-empty boxes| R1
  FOLD32 -->|non-empty boxes| POOL32
  FOLD8 -->|non-empty boxes| POOL8
  POOL32 --> NG
  POOL32 --> B32
  POOL8 --> B8
  MLP -->|top-2| OUT
  CNN32 -->|Grad-CAM, top-2| OUT
  CNN8 -->|top-2| OUT
  R1 --> OUT
  NG --> OUT
  B32 --> OUT
  B8 --> OUT
  style B8 stroke-width:3px
```

Everything runs on the CPU on 127.0.0.1. Models load once at startup from weights/*.pt, weights/ngram_lm.pkl, weights/bert_int8_dynamic.pt and the Hugging Face cache (bert-base-uncased).
The app and the benchmark call the same functions (`data/word_pipeline.py` for strip -> box images -> probabilities in the
benchmark and box images -> probabilities in the app,
`models/ladder.py` for the correction rows); the diagram and slide 4 are generated from `scripts/architecture.py`.

## Results (R4)

CPU timing only: AMD Ryzen AI 9 HX 370 w/ Radeon 890M, 4 torch threads, 24 logical cores, 31.1 GB RAM; GPU present but unused for timing: NVIDIA GeForce RTX 5070 Ti Laptop GPU.
Windows-11-10.0.26300-SP0, Python 3.13.15, torch 2.9.1+cu128, transformers 5.23.0. Timing on AC power, power scheme Turbo, power mode Best performance.
Timing protocol: Table 1 CPU ms and the standalone BERT ms are the mean of 100 timed runs after 10 warm-up runs; Table 2 ms/word is the end-to-end mean over the 300 test worksheets; the median of 3 timing passes (by app-model latency) is reported.

### Table 1 - Vision (EMNIST balanced test, N=18800)

| Model | 47-class acc % | 36-folded acc % | Disk MB | CPU b1 ms | CPU b8 ms |
|---|---|---|---|---|---|
| dense_mlp_fp32 | 86.41 | 88.34 | 2.08 | 0.050 | 0.065 |
| char_cnn_fp32 | 89.47 | 91.32 | 6.53 | 0.283 | 0.805 |
| char_cnn_int8 | 89.43 | 91.29 | 1.65 | 0.254 | 0.361 |

### Table 2 - Word ladder (N=300 held-out worksheets, boxed strips)

| Pipeline | Word acc % | Char acc % | Size MB | Mean ms/word |
|---|---|---|---|---|
| (1) Trained NN, no correction: FP32 CNN | 61.67 | 90.14 | 6.5 | 4.9 |
| (1b) parity: INT8 CNN, no correction | 60.67 | 89.95 | 1.7 | 3.7 |
| (2) Classical: FP32 CNN + SymSpell + n-gram fusion | 86.33 | 91.17 | 25.5 | 10.7 |
| (3) Transformer: FP32 CNN + same pool + FP32 BERT | 88.00 | 92.46 | 443.4 | 165.0 |
| (4) Quantized Transformer [App]: INT8 CNN pool + INT8 BERT | 87.67 | 91.43 | 216.2 | 49.9 |
| (ablation) FP32 CNN + same pool + INT8 BERT | 87.33 | 91.37 | 221.1 | 54.6 |

Candidate recall (target in the shared pool): FP32 CNN 88.67 %, INT8 CNN 89.00 %; this is the ceiling for
rows 2–4. Strip-path character accuracy 90.14 % vs raw EMNIST 91.32 % (gap +1.18 pp). All five checks pass
(`results/benchmark_results.md`). Test OOV rate (target not in the lexicon): 12.7 %
(`results/corpus_sources.json`).

Noise: with N=300 test worksheets one item is 0.33 pp of word accuracy, so rows 3, 4 and the ablation (88.00 /
87.67 / 87.33 %) differ by one or two items and are within noise of each other; the gaps to rows 1 and 2 are
not. Row 2 is the classical correction baseline: SymSpell + n-gram fusion run on the same trained CNN's output,
because a classical method with no trained reader cannot turn pixels into letters, and sharing the CNN and the
candidate pool isolates the correction technique as the only thing that changes between rows 2, 3 and 4.

Reference, not trained by us (`results/vlm_reference.json`, recomputed by `scripts/recompute_metrics.py`):
Qwen3-VL 8B (6.2 GB, Q4_K_M) read the same 300 strips zero-shot on the laptop GPU (NVIDIA GeForce RTX 5070 Ti Laptop GPU), with no
tuning. From the strip alone it scores 80.67 % word / 92.91 % char accuracy
(compare row 1); given the sentence too, 89.33 % / 94.46 % (compare rows 3 and 4),
at about 966 ms/word on the GPU. This timing is not comparable
with the CPU column above. It has no pass/fail check and is not part of the ladder: it shows what a model ~29x the
size of our app model reaches on the same task.

Latencies above were re-timed twice on AC power in Windows "Best performance" mode (three timing passes each,
median reported; every prediction verified identical to the accuracy run). The first re-timing replaced run 3's
throttled timings (kept in `results/benchmark_run3_timing_superseded.json`); the second followed the
48-WordPiece BERT context-window change, with outputs again verified identical (replaced timings in
`results/benchmark_retime1_superseded.json`).

### Methodology

- All fusion weights were tuned on the 50 dev worksheets only (EMNIST validation glyphs the CNN never trained
  on), with a continuous objective (mean log-softmax of the target over the candidate pool) on a simplex.
- The 300 test worksheets (EMNIST test glyphs) were evaluated three times. Run 3 is reported. Run 1 was
  replaced because its tuner pushed every weight to the edge of its grid; run 2 because the candidate pool
  was cut with the fusion weights and training read the test file for an out-of-vocabulary
  statistic. Both fixes were made on the code and dev side only; reasons and the old numbers are in
  `results/benchmark_run1_superseded.md` and `results/benchmark_run2_superseded.md`.
- `results/test_predictions.json` holds every per-item prediction; `python scripts/recompute_metrics.py`
  re-derives all reported numbers from it without torch.

## Reproduction

```
python -m data.prepare_data                 # EMNIST download (~536 MB, MD5-checked), splits, worksheets
python scripts/train_vision.py              # Dense MLP, CharCNN FP32, CharCNN INT8 (a GPU speeds this up)
python scripts/train_ngram.py               # n-gram LM, dev fusion tuning -> results/ngram_metrics.json
python scripts/tune_bert_fusion.py          # BERT fusion weights on dev -> results/bert_fusion_params.json
python scripts/evaluate_all.py              # Tables 1 and 2 (CPU; run alone) -> results/benchmark_results.*
python scripts/evaluate_all.py --retime     # timing-only re-measure; outputs must match the committed predictions
python scripts/generate_slides.py --name "Abdalraouf Alkabir" --student-id 2309015482 --video-url "..."
```

`python scripts/evaluate_all.py --worksheets data/worksheets_dev_50.json --no-save` runs the benchmark on
the dev set without writing anything.

## Limitations

- Digit strings read with letters among the digits (`20O0` for 2000) never enter the candidate pool: the
  lexicon holds words only, and a digit string is a candidate only when every box reads as a digit.
- Span pseudo-log-likelihood favours multi-piece words, because the other pieces stay visible
  (`REDUCED` -> `REJOICED` in `results/failure_examples.json`).
- 12.7 % of test targets are not in the lexicon, and only 88–89 % of targets reach the candidate pool.
- BERT scores use at most 48 WordPieces on each side of the blank; the benchmark sentences are all shorter, so this does not affect reported results.
- Full-page OCR returns plain text only (no field extraction yet), and the app still needs the answer reader's
  weights to start, even when only OCR is used.
- Next steps: digit-mapped numeric candidates, PLL-word-l2r scoring, a larger lexicon; OCR field extraction
  and an OCR-only start.

## Security notes

The app binds to 127.0.0.1 only, accepts only `localhost`/`127.0.0.1` Host headers and JSON request bodies,
caps the body at 8 MB and each image at 5 MB / 4096×4096 (checked before decoding), and returns one-line
errors without tracebacks. Recognition runs one request at a time. `weights/ngram_lm.pkl` is a pickle: it is
installed only through `scripts/download_assets.py`, which checks its sha256, and it is read with a restricted
unpickler that only allows the numpy types the model needs, so a tampered file cannot run code.
bert-base-uncased is loaded at the pinned Hugging Face revision `86b5e0934494bd15c9632b12f734a8a67f723594`.

## Citations

- G. Cohen, S. Afshar, J. Tapson, A. van Schaik. *EMNIST: an extension of MNIST to handwritten letters.* 2017.
- J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. *BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding.* NAACL 2019 (`bert-base-uncased`).
- Project Gutenberg texts (public domain) and Wikipedia articles (CC BY-SA 4.0); every source is listed in
  `results/corpus_sources.json`.

## License

- The code in this repository is released under the MIT License (see [LICENSE](LICENSE)).
- The model weights in the release assets (`assets-v1`, `assets-v2`) are released under the MIT License as trained artifacts of this project.
- Third-party terms still apply to what they cover:
  - `bert-base-uncased` (Google): Apache License 2.0.
  - EMNIST (NIST): cite G. Cohen, S. Afshar, J. Tapson, A. van Schaik, *EMNIST: an extension of MNIST to handwritten letters*, 2017.
  - Project Gutenberg texts in the `assets-v1` corpus cache: public domain in the US; the Project Gutenberg license applies to their headers.
  - Wikipedia text in the `assets-v1` corpus cache: CC BY-SA 4.0 with attribution (sources in `results/corpus_sources.json`); share-alike applies to that text, not to the code.

## AI usage

Built by Abdalraouf Alkabir with Claude (Anthropic) as the AI coding assistant.
Student: chose the use case, made the design and evaluation decisions, ran all training and benchmarks on own laptops, reviewed the results.
Claude: generated and reviewed code and tests, drafted the README and slides.
All numbers come from the student's own runs (results/*.json); scripts/recompute_metrics.py re-derives them without the models.
Pretrained model: bert-base-uncased (Devlin et al., 2019). Data: EMNIST (Cohen et al., 2017), Project Gutenberg, Wikipedia (CC BY-SA).

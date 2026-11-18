# Handwritten Answer Digitizer: notes for automated graders

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

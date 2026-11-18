# Benchmark run 2 (superseded)

This was the second benchmark run over the 300 test worksheets; it is kept for transparency. The reported result is `results/benchmark_results.md` (run 3); run 1 is in `results/benchmark_run1_superseded.md`.

## Why it was superseded

A code review of the run-2 code found two problems on the code and dev side. First, the classical candidate pool was cut to 20 words using the fusion weights, so the classical tuner optimised over a pool built with the default weights, while inference, BERT tuning and the benchmark cut the pool with the tuned weights. Second, `scripts/train_ngram.py` read the test worksheets (after tuning) to report an out-of-vocabulary statistic. Both were fixed in code and on the dev set only. The pool is now cut by a fixed rule (visual log-probability minus 2.0 x weighted edit distance) that never reads the fusion weights, and the training pipeline never opens the test file. The classical and BERT fusion weights were re-tuned on the dev worksheets. No change was informed by test results.

## Run 2 results as reported

CPU timing only: AMD Ryzen AI 9 HX 370 w/ Radeon 890M, 4 torch threads, 24 logical cores, 31.1 GB RAM; GPU present but unused for timing: NVIDIA GeForce RTX 5070 Ti Laptop GPU.
Windows-11-10.0.26300-SP0, Python 3.13.15, torch 2.9.1+cu128, transformers 5.23.0.

## Table 1 - Vision (EMNIST balanced test, N=18800)

| Model | 47-class acc % | 36-folded acc % | Disk MB | CPU b1 ms | CPU b8 ms |
|---|---|---|---|---|---|
| dense_mlp_fp32 | 86.41 | 88.34 | 2.08 | 0.067 | 0.121 |
| char_cnn_fp32 | 89.47 | 91.32 | 6.53 | 0.835 | 2.093 |
| char_cnn_int8 | 89.43 | 91.29 | 1.65 | 0.615 | 0.878 |

## Table 2 - Word ladder (N=300 worksheets from worksheets_test_300.json, boxed strips)

| Pipeline | Word acc % | Char acc % | Size MB | Mean ms/word |
|---|---|---|---|---|
| (1) Trained NN, no correction: FP32 CNN | 61.67 | 90.14 | 6.5 | 8.1 |
| (1b) parity: INT8 CNN, no correction | 60.67 | 89.95 | 1.7 | 6.5 |
| (2) Classical: FP32 CNN + SymSpell + n-gram fusion | 86.33 | 91.17 | 25.5 | 18.5 |
| (3) Transformer: FP32 CNN + same pool + FP32 BERT | 87.67 | 92.33 | 443.4 | 301.3 |
| (4) Quantized Transformer [App]: INT8 CNN + INT8 BERT | 88.00 | 92.20 | 216.2 | 109.9 |
| (ablation) FP32 CNN + same pool + INT8 BERT | 87.67 | 92.07 | 221.1 | 115.1 |

Candidate recall (target in the shared pool): FP32 CNN 88.00 %, INT8 CNN 88.33 % (ceiling for rows 2-4).
Strip-path char accuracy 90.14 % vs raw EMNIST 36-folded 91.32 %: gap +1.18 pp (within 5.0 pp)
Sizes sum the weight files each pipeline loads (FP32 BERT: serialized state_dict; INT8 BERT: saved state_dict file; the ~0.2 MB tokenizer vocab is not counted). Row (4) simulates the app pipeline offline: the same INT8 artifacts, measured here, not inside the app.
Standalone BERT: FP32 417.8 MB, 46.1 ms; INT8 195.6 MB, 16.4 ms (one masked sentence, batch 1).

## Checks (reported, not enforced)

| Check | Result | Value |
|---|---|---|
| Raw < N-gram < BERT (word acc) | PASS | 61.67 / 86.33 / 87.67 % |
| INT8 BERT size reduction >= 2.0x | PASS | 2.14x |
| INT8 BERT word-acc drop <= 1.5 pp | PASS | +0.00 pp |
| INT8 CNN 36-class acc drop <= 1.5 pp | PASS | +0.03 pp |

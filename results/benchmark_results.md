# Benchmark results

CPU timing only: AMD Ryzen AI 9 HX 370 w/ Radeon 890M, 4 torch threads, 24 logical cores, 31.1 GB RAM; GPU present but unused for timing: NVIDIA GeForce RTX 5070 Ti Laptop GPU.
Windows-11-10.0.26300-SP0, Python 3.13.15, torch 2.9.1+cu128, transformers 5.23.0. Timing on AC power, power scheme Turbo, power mode Best performance.
Table 1 and the standalone BERT timings are the mean of 100 timed runs after 10 warm-up runs. Table 2 is the offline pipeline mean over the 300 test worksheets with known box counts. Timing includes strip slicing and preprocessing. HTTP, the browser UI, and Grad-CAM are outside the timed pipeline. The median of 3 timing passes by app-model latency is reported.

## Table 1 - Vision (EMNIST balanced test, N=18800)

| Model | 47-class acc % | 36-folded acc % | Disk MiB | CPU b1 ms | CPU b8 ms |
|---|---|---|---|---|---|
| dense_mlp_fp32 | 86.41 | 88.34 | 2.08 | 0.050 | 0.065 |
| char_cnn_fp32 | 89.47 | 91.32 | 6.53 | 0.283 | 0.805 |
| char_cnn_int8 | 89.43 | 91.29 | 1.65 | 0.254 | 0.361 |

## Table 2 - Word ladder (N=300 worksheets from worksheets_test_300.json, boxed strips)

| Pipeline | Word acc % | Char acc % | Size MiB | Mean ms/word |
|---|---|---|---|---|
| (1) Trained NN, no correction: FP32 CNN | 61.67 | 90.14 | 6.5 | 4.9 |
| (1b) parity: INT8 CNN, no correction | 60.67 | 89.95 | 1.7 | 3.7 |
| (2) Classical: FP32 CNN + SymSpell + n-gram fusion | 86.33 | 91.17 | 25.5 | 10.7 |
| (3) Transformer: FP32 CNN + same pool + FP32 BERT | 88.00 | 92.46 | 443.4 | 165.0 |
| (4) Quantized Transformer [App]: INT8 CNN pool + INT8 BERT | 87.67 | 91.43 | 216.2 | 49.9 |
| (ablation) FP32 CNN + same pool + INT8 BERT | 87.33 | 91.37 | 221.1 | 54.6 |

Candidate recall (target in the shared pool): FP32 CNN 88.67 %, INT8 CNN 89.00 % (ceiling for rows 2-4).
Strip-path char accuracy 90.14 % vs raw EMNIST 36-folded 91.32 %: gap +1.18 pp (within 5.0 pp)
Sizes sum the model weights loaded by each pipeline, including each BERT's serialized state_dict. Sizes exclude the tokenizer vocabulary. Row 4 uses the same candidate rule on the INT8 CNN's output. Rows 2, 3 and the ablation share the pool from the FP32 CNN.
Standalone BERT (one masked sentence, batch 1). FP32 417.8 MiB, 25.5 ms. INT8 195.6 MiB, 8.5 ms.

## Checks (reported, not enforced)

| Check | Result | Value |
|---|---|---|
| Raw < N-gram < BERT FP32 (word acc) | PASS | 61.67 / 86.33 / 88.00 % |
| Raw < N-gram < quantized app (word acc) | PASS | 61.67 / 86.33 / 87.67 % |
| INT8 BERT size reduction >= 2.0x | PASS | 2.14x |
| INT8 BERT word-acc drop <= 1.5 pp | PASS | +0.67 pp |
| INT8 CNN 36-class acc drop <= 1.5 pp | PASS | +0.03 pp |

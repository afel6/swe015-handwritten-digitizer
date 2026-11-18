# Benchmark results

CPU timing only: AMD Ryzen AI 9 HX 370 w/ Radeon 890M, 4 torch threads, 24 logical cores, 31.1 GB RAM; GPU present but unused for timing: NVIDIA GeForce RTX 5070 Ti Laptop GPU.
Windows-11-10.0.26300-SP0, Python 3.13.15, torch 2.9.1+cu128, transformers 5.23.0.

## Table 1 - Vision (EMNIST balanced test, N=18800)

| Model | 47-class acc % | 36-folded acc % | Disk MB | CPU b1 ms | CPU b8 ms |
|---|---|---|---|---|---|
| dense_mlp_fp32 | 86.41 | 88.34 | 2.08 | 0.132 | 0.177 |
| char_cnn_fp32 | 89.47 | 91.32 | 6.53 | 1.828 | 4.266 |
| char_cnn_int8 | 89.43 | 91.29 | 1.65 | 6.270 | 5.452 |

## Table 2 - Word ladder (N=300 worksheets from worksheets_test_300.json, boxed strips)

| Pipeline | Word acc % | Char acc % | Size MB | Mean ms/word |
|---|---|---|---|---|
| (1) Trained NN, no correction: FP32 CNN | 61.67 | 90.14 | 6.5 | 13.4 |
| (1b) parity: INT8 CNN, no correction | 60.67 | 89.95 | 1.7 | 11.6 |
| (2) Classical: FP32 CNN + SymSpell + n-gram fusion | 86.33 | 91.17 | 25.5 | 26.4 |
| (3) Transformer: FP32 CNN + same pool + FP32 BERT | 88.00 | 92.46 | 443.4 | 385.3 |
| (4) Quantized Transformer [App]: INT8 CNN pool + INT8 BERT | 87.67 | 91.43 | 216.2 | 176.9 |
| (ablation) FP32 CNN + same pool + INT8 BERT | 87.33 | 91.37 | 221.1 | 182.9 |

Candidate recall (target in the shared pool): FP32 CNN 88.67 %, INT8 CNN 89.00 % (ceiling for rows 2-4).
Strip-path char accuracy 90.14 % vs raw EMNIST 36-folded 91.32 %: gap +1.18 pp (within 5.0 pp)
Sizes sum the weight files each pipeline loads (both BERTs: serialized state_dict of the model loaded for the run; the ~0.2 MB tokenizer vocab is not counted). Row (4) is the app pipeline measured offline: its pool uses the same fixed rule on the INT8 CNN's own read, so it can differ from the FP32-CNN pool of rows 2, 3 and the ablation.
Standalone BERT: FP32 417.8 MB, 70.6 ms; INT8 195.6 MB, 24.0 ms (one masked sentence, batch 1).

## Checks (reported, not enforced)

| Check | Result | Value |
|---|---|---|
| Raw < N-gram < BERT FP32 (word acc) | PASS | 61.67 / 86.33 / 88.00 % |
| Raw < N-gram < quantized app (word acc) | PASS | 61.67 / 86.33 / 87.67 % |
| INT8 BERT size reduction >= 2.0x | PASS | 2.14x |
| INT8 BERT word-acc drop <= 1.5 pp | PASS | +0.67 pp |
| INT8 CNN 36-class acc drop <= 1.5 pp | PASS | +0.03 pp |

# Benchmark run 1 (superseded)

This is the first benchmark run over the 300 test worksheets. It is kept for transparency.
The reported result is `results/benchmark_results.md` (run 2).

## Why it was superseded

The first fusion tuner used a free grid with the dev objective (mean log-softmax of the target
among the candidate scores). That objective always improves when every weight is scaled up once
the target already ranks first. On the dev set it pushed every fused weight to the edge of its
grid, the dev objective saturated at about -1e-5, and classical dev word accuracy fell from
0.92 to 0.88. A code review found this, using dev results only. The fix also used dev only: the
fused weights were constrained to a simplex (they sum to 1), so the score scale can no longer be
inflated. Both correctors were re-tuned on dev, and the test benchmark was run exactly once more.
Nothing was tuned on test results. Run 2 is lower than or equal to run 1 on every corrected row.

## Run 1 - Table 2, word ladder (N=300 test worksheets, boxed strips)

| Pipeline | Word acc % | Char acc % | Size MB | Mean ms/word |
|---|---|---|---|---|
| (1) Trained NN, no correction: FP32 CNN | 61.67 | 90.14 | 6.5 | 8.3 |
| (1b) parity: INT8 CNN, no correction | 60.67 | 89.95 | 1.7 | 6.6 |
| (2) Classical: FP32 CNN + SymSpell + n-gram fusion | 87.67 | 92.53 | 25.5 | 18.5 |
| (3) Transformer: FP32 CNN + same pool + FP32 BERT | 88.33 | 92.40 | 443.4 | 305.6 |
| (4) Quantized Transformer [App]: INT8 CNN + INT8 BERT | 88.67 | 92.53 | 216.2 | 112.2 |
| (ablation) FP32 CNN + same pool + INT8 BERT | 88.33 | 92.46 | 221.1 | 119.6 |

Candidate recall (target in the shared pool): FP32 CNN 88.67 %, INT8 CNN 89.00 %.

Checks: Raw < N-gram < BERT PASS (61.67 / 87.67 / 88.33 %); INT8 BERT size reduction PASS (2.14x);
INT8 BERT word-acc drop PASS (+0.00 pp); INT8 CNN 36-class acc drop PASS (+0.03 pp).

## Run 1 - tuned parameters (free grid, dev only)

| Corrector | Parameters | Dev objective | Dev word acc |
|---|---|---|---|
| Classical n-gram fusion | vis_weight 0.5, edit_penalty 4.0, lm_weight 0.25 (grid vis 0.5-2, edit 0.5-4, lm 0.25-2) | -0.0304 | 0.88 |
| BERT fusion (FP32) | alpha 0.0, lambda_vis 2.0, lambda_bert 1.0, gamma 4.0 (grid alpha 0-1, vis 0.25-2, bert 0.25-2, gamma 0-4) | -1.4e-05 | 0.90 |
| BERT fusion (INT8 alone) | alpha 0.0, lambda_vis 2.0, lambda_bert 2.0, gamma 4.0 | -4.5e-05 | - |

Every classical weight and three of the four BERT parameters sit on a grid edge.

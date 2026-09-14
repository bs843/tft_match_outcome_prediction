# Model-selection audit

Status: **PASS**

The top-four winner was locked at `2026-08-25T19:11:06+00:00` before the
test partition was first accessed at `2026-08-25T19:11:06+00:00`.
Final-test results were excluded from selection.

## Top-four classification

- Selected model: `xgboost`
- Primary metric: validation F1
- CV F1: 0.849 ± 0.004
- Validation F1: 0.854
- Runner-up: `lightgbm` (0.853)
- Margin: 0.0018
- Within observed CV variability: True

The materiality statement is descriptive: Margin is compared with the larger observed five-fold CV standard deviation of the winner and runner-up; this is a descriptive stability comparison, not a significance threshold.

## Exact placement

- Selected model: `random_forest`
- Validation MAE: 1.107
- Runner-up: `dummy` (3.500)
- MAE improvement: 2.393

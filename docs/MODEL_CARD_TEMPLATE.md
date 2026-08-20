# Model Card

## Intended use

Offline dissertation analysis of completed patch-17.8 TFT matches. Not for live-game advice, causal inference, player ranking, scouting or deployment.

## Training data

Record dataset version, collection run ID, accepted-match count, patch window, sampling strata, schema version, raw checksum-set hash and split counts.

## Features and target

Document the top-four target, enabled composition families, training-only vocabulary threshold, and prohibited leakage fields.

## Model selection

Record all candidate specifications, grouped cross-validation scores, validation metrics, selected hyperparameters and random seed.

## Test results

Report ROC AUC, log loss, Brier score, accuracy, F1, expected calibration error, within-match top-four hit rate, placement Spearman correlation and match-cluster bootstrap intervals.

## Limitations

Discuss post-match feature timing, selection bias, repeated players, patch specificity, API missingness, rank-sampling bias, non-causal interpretation and lack of live-game validity.

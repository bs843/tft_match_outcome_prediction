# Thesis results summary

## Dataset

The frozen Patch 17.8 dataset contains 5,000 accepted matches and 40,000
participant end-state observations. The authoritative chronological match-group
split contains 3,500 train, 750 validation and 750 test matches. The source
feature table contains 291 predictors; train-fitted
F5 encoding produces 404 model columns. The outcome
is exactly balanced because every valid eight-player match contributes four
top-four and four bottom-four observations.

## Model comparison

Six frozen classifiers were compared. Five-fold CV was available for F1 only.
The highest mean CV F1 was `lightgbm`;
the validation winner and pre-test-selected production classifier was XGBoost.
XGBoost validation F1 was 0.854; the runner-up was
`lightgbm` at 0.853, a margin of
0.0018. This margin is smaller than the observed fold-level CV
standard deviation reference (0.0058), so the
leading models should be described as close rather than decisively separated.

## Final test performance

The locked XGBoost achieved F1 0.854
(95% match-cluster bootstrap CI 0.845–0.862),
accuracy 0.853, precision 0.852,
recall 0.856, ROC-AUC 0.930, and
PR-AUC 0.933. The exact-placement random forest achieved
test MAE 1.065 and accuracy 0.349.

## Generalisation

XGBoost CV mean F1 was 0.849, validation F1 was
0.854, and test F1 was 0.854.
The CV→validation gap was +0.0049, the
validation→test gap was -0.0006, and the
overall CV→test gap was +0.0043. These small signed
gaps indicate stable aggregate performance across the frozen evaluation stages.

## Interpretation

Permutation importance used the train-fitted development XGBoost on validation;
it did not alter selection. The largest aggregate predictive-contribution group
was `composition_structure`. The highest individual columns were:
`structure__total_items`, `structure__mean_star`, `level`, `item_count__tft_item_thiefsgloves`, `structure__three_star_units`. These are
associations with model output, not causal effects.

## Error analysis

The final confusion matrix contains 2,553 true negatives, 447 false positives,
433 false negatives and 2,567 true positives. Error rates for the natural
placement boundary were 0.368 in fourth place and 0.380 in fifth
place. Rank robustness covers 9 strata and region robustness covers
5 platform regions; sample sizes accompany every estimate.

## Research objectives

### Aim 1

The descriptive tables show outcome-associated differences in champion, trait,
item, player-level and composition-structure variables. The strongest patterns
should be interpreted alongside `feature_group_importance.csv` and the outcome
frequency tables, without causal language or indiscriminate significance testing.

### Aim 2

The frozen end-state F5 feature set predicted top-four placement with final-test
F1 0.854 and ROC-AUC 0.930. This
supports substantial predictive information in final-board state while leaving
meaningful uncertainty and classification error.

## Warnings

- Fold-level CV stores F1 only; CV variability for other metrics is unavailable.
- Predicted probabilities were not persisted and will be reproduced from the frozen model and matrix.
- Source predictor count is 291; train-fitted F5 encoding contains 404 model columns.
- SHAP unavailable or incompatible; skipped without affecting other outputs: ModuleNotFoundError: No module named 'shap'
- Rank strata with fewer than 50 test observations are retained but should not be interpreted as stable estimates: BRONZE, CHALLENGER, GRANDMASTER
- Calibration intercept/slope were not fitted on the final test set; Brier score and observed-versus-predicted bins are reported without recalibration.

# Thesis discussion support

## Strongest finding

- The locked XGBoost generalised with test F1 0.854; see
  `tables/final_test_model_summary.csv` and `tables/final_test_metric_ci.csv`.

## Baseline versus complex models

- The dummy classifier had validation F1 0.000, while the leading boosted models
  were close to 0.85; see `tables/validation_model_summary.csv`.
- The narrow XGBoost–lightgbm margin is within observed CV variability;
  avoid claiming decisive superiority.

## Feature interpretation

- `composition_structure` had the largest aggregated absolute
  permutation importance; see `tables/feature_group_importance.csv`.
- Permutation and native importance describe model reliance, not causal effects.

## Generalisation

- CV→test F1 changed by +0.0043; see
  `tables/generalisation_gaps.csv`.

## Rank and region robustness

- Compare estimates only with their sample sizes in `tables/test_metrics_by_rank.csv`
  and `tables/test_metrics_by_region.csv`.
- Subgroup differences are descriptive; no subgroup was used as a predictor.

## TFT-specific interpretation and randomness

- End-state boards encode accumulated strategic decisions and stochastic shop,
  item and combat outcomes. Predictive accuracy does not isolate these mechanisms.

## Limitations

- Patch-specific: conclusions apply to TFT Patch 17.8 and Set 17.
- End-state data: this is outcome association from final boards, not live-game prediction.
- API/data availability: the sample is limited to the frozen Riot API collection window.
- Model: fixed baseline parameters were used; the test set was not a tuning resource.
- External validity: rank and region checks are internal strata, not independent replications.

## Future work

- Replicate the frozen protocol on a later patch as a separately versioned study.
- Study temporal game-state trajectories only under a new, pre-specified design.
- Evaluate calibration and subgroup uncertainty on independent future data.

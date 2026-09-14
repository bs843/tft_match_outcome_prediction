# Uncertainty and final-evaluation addendum

All intervals use 2,000 deterministic bootstrap resamples of complete matches. They are
descriptive uncertainty intervals; no result in this addendum changes model, feature, or
threshold selection.

- The strongest substantive predictive unit is `structure`: its removal
  reduced F1 by 0.0246 on average
  (paired-fold 95% t interval 0.0184 to 0.0309);
  all 5 folds deteriorated.
- Frozen logistic feature-family validation F1 was 0.8494 for F4 and 0.8511
  for F5, a difference of 0.0017; this supports “F4 nearly equivalent,”
  not a claim of statistically established superiority.
- `top4_match_bootstrap_ci.csv` reports validation and test uncertainty for discrimination,
  classification, and probability metrics.
- `placement_match_bootstrap_ci.csv` reports final-test uncertainty for the already locked
  placement model; the confusion matrix and per-placement table expose where moderate exact
  prediction remains difficult.

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import write_text
from .verify import VerifiedInputs


def write_thesis_reports(verified: VerifiedInputs, output: Path,
                         performance: dict[str, Any], diagnostics: dict[str, Any],
                         robustness: dict[str, pd.DataFrame],
                         interpretation: dict[str, Any], descriptive: dict[str, Any],
                         warnings: list[str]) -> None:
    selected = performance["performance"].query("selected").iloc[0]
    audit = performance["audit"]["top_four"]
    ci = diagnostics["bootstrap"].set_index("metric")
    groups = interpretation["groups"]
    top_features = interpretation["permutation"].head(10)
    placement_errors = pd.read_csv(output / "tables" / "errors_by_final_placement.csv")
    fourth = placement_errors.loc[placement_errors["placement"].eq(4), "error_rate"].iloc[0]
    fifth = placement_errors.loc[placement_errors["placement"].eq(5), "error_rate"].iloc[0]
    rank = robustness["rank"]
    region = robustness["region"]
    summary = f"""# Thesis results summary

## Dataset

The frozen Patch 17.8 dataset contains 5,000 accepted matches and 40,000
participant end-state observations. The authoritative chronological match-group
split contains 3,500 train, 750 validation and 750 test matches. The source
feature table contains {verified.source_predictor_count} predictors; train-fitted
F5 encoding produces {verified.encoded_feature_count} model columns. The outcome
is exactly balanced because every valid eight-player match contributes four
top-four and four bottom-four observations.

## Model comparison

Six frozen classifiers were compared. Five-fold CV was available for F1 only.
The highest mean CV F1 was `{performance['cv'].groupby('model')['f1'].mean().idxmax()}`;
the validation winner and pre-test-selected production classifier was XGBoost.
XGBoost validation F1 was {selected['validation_f1']:.3f}; the runner-up was
`{audit['runner_up']}` at {audit['runner_up_validation_f1']:.3f}, a margin of
{audit['margin']:.4f}. This margin is smaller than the observed fold-level CV
standard deviation reference ({audit['cv_variability_reference']:.4f}), so the
leading models should be described as close rather than decisively separated.

## Final test performance

The locked XGBoost achieved F1 {selected['test_f1']:.3f}
(95% match-cluster bootstrap CI {ci.loc['f1','lower_95']:.3f}–{ci.loc['f1','upper_95']:.3f}),
accuracy {selected['test_accuracy']:.3f}, precision {selected['test_precision']:.3f},
recall {selected['test_recall']:.3f}, ROC-AUC {selected['test_roc_auc']:.3f}, and
PR-AUC {selected['test_pr_auc']:.3f}. The exact-placement random forest achieved
test MAE 1.065 and accuracy 0.349.

## Generalisation

XGBoost CV mean F1 was {selected['cv_f1_mean']:.3f}, validation F1 was
{selected['validation_f1']:.3f}, and test F1 was {selected['test_f1']:.3f}.
The CV→validation gap was {selected['cv_to_validation_f1_gap']:+.4f}, the
validation→test gap was {selected['validation_to_test_f1_gap']:+.4f}, and the
overall CV→test gap was {selected['cv_to_test_f1_gap']:+.4f}. These small signed
gaps indicate stable aggregate performance across the frozen evaluation stages.

## Interpretation

Permutation importance used the train-fitted development XGBoost on validation;
it did not alter selection. The largest aggregate predictive-contribution group
was `{groups.iloc[0]['feature_group']}`. The highest individual columns were:
{', '.join(f'`{name}`' for name in top_features['feature'].head(5))}. These are
associations with model output, not causal effects.

## Error analysis

The final confusion matrix contains 2,553 true negatives, 447 false positives,
433 false negatives and 2,567 true positives. Error rates for the natural
placement boundary were {fourth:.3f} in fourth place and {fifth:.3f} in fifth
place. Rank robustness covers {len(rank)} strata and region robustness covers
{len(region)} platform regions; sample sizes accompany every estimate.

## Research objectives

### Aim 1

The descriptive tables show outcome-associated differences in champion, trait,
item, player-level and composition-structure variables. The strongest patterns
should be interpreted alongside `feature_group_importance.csv` and the outcome
frequency tables, without causal language or indiscriminate significance testing.

### Aim 2

The frozen end-state F5 feature set predicted top-four placement with final-test
F1 {selected['test_f1']:.3f} and ROC-AUC {selected['test_roc_auc']:.3f}. This
supports substantial predictive information in final-board state while leaving
meaningful uncertainty and classification error.

## Warnings

{chr(10).join(f'- {warning}' for warning in warnings) if warnings else '- None'}
"""
    write_text(output / "thesis_results_summary.md", summary)
    discussion = f"""# Thesis discussion support

## Strongest finding

- The locked XGBoost generalised with test F1 {selected['test_f1']:.3f}; see
  `tables/final_test_model_summary.csv` and `tables/final_test_metric_ci.csv`.

## Baseline versus complex models

- The dummy classifier had validation F1 0.000, while the leading boosted models
  were close to 0.85; see `tables/validation_model_summary.csv`.
- The narrow XGBoost–{audit['runner_up']} margin is within observed CV variability;
  avoid claiming decisive superiority.

## Feature interpretation

- `{groups.iloc[0]['feature_group']}` had the largest aggregated absolute
  permutation importance; see `tables/feature_group_importance.csv`.
- Permutation and native importance describe model reliance, not causal effects.

## Generalisation

- CV→test F1 changed by {selected['cv_to_test_f1_gap']:+.4f}; see
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
"""
    write_text(output / "thesis_discussion_points.md", discussion)

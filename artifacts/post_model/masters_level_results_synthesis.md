# Master's-level results synthesis

## 1. Research-question mapping

| RQ | Analysis | Artifact | Core finding |
|---|---|---|---|
| RQ1 | Support-filtered descriptive association | `feature_evidence_table.csv` | Unit, item, trait-signature and unit-star outcomes vary materially in the frozen cohort. |
| RQ2 | Training-only adjusted logistic synthesis | `raw_vs_adjusted_associations.csv` | Some raw associations persist; others attenuate or change direction after selected-covariate adjustment. |
| RQ3 | Frozen F1–F5 validation experiment | `feature_family_incremental_value.csv` | The largest transition was F1 → F2 (+0.157 F1). |
| RQ4 | Locked XGBoost validation TreeSHAP | `shap_vs_descriptive.csv` | Predictive importance and raw uplift are related but non-interchangeable quantities. |
| RQ5 | Frozen secondary placement evaluation | `../thesis_extension_v1/tables/placement_final_test_metrics.csv` | Exact placement is better than the dummy baseline but materially harder than top-four classification. |
| RQ6 | Threshold and development-partition sensitivity | `support_threshold_sensitivity.csv` | Ranking and directional stability are reported without using test outcomes for discovery. |

## 2. Composition associations

Shen, Force Of Nature and the Space Groove tier-4 / seven-unit signature were among the strongest supported positive descriptive findings. Aurora was a leading negative unit association. These are final-board associations, not recommended interventions.

## 3. Adjustment

The integrated table distinguishes persisted, attenuated and directionally unstable findings. Adjustment is restricted to the frozen training partition and selected observed covariates; residual endogeneity and survivorship remain.

## 4. Predictive contribution

Units/star levels created the largest cumulative improvement over F1. Items added a further substantial increment at F3 → F4, while F4 → F5 added approximately 0.002 F1. Incremental prediction is not causal importance.

## 5. Model interpretation

TreeSHAP explains the locked XGBoost's validation predictions. Total item count, mean star level and player level were major model contributions. High SHAP importance can coexist with modest raw uplift because the model uses conditional nonlinear patterns.

## 6. Robustness

Predeclared support thresholds were 100, 200, 300 and 500 for main features. The retained interaction artifact begins at n=150, so interaction sensitivity was evaluated at 150, 250 and 500; n=100 cannot be reconstructed without rerunning the exploratory search. Across 182 supported development comparisons, 87.4% retained the same train/validation direction. BH summaries separate raw p-values from multiplicity-adjusted evidence.

## 7. Exact-placement secondary analysis

The locked random forest achieved MAE 1.065, median absolute error 1.0, 34.9% exact accuracy, 72.4% within one place and 89.9% within two, versus dummy MAE 3.5. No new test evaluation was performed.

## 8. Limitations

Survivorship, endogeneity, item acquisition, upgrade economics, retrieval-time rank, missing decision trajectories, stochasticity and patch specificity constrain interpretation. Augments were absent from the frozen schema and were not reconstructed.

## 9. Overall answer

The project supports a coherent progression from descriptive association, through selected-covariate adjustment, to incremental predictive contribution and frozen-model interpretation. It does not identify causal effects of choosing a unit, item or trait.

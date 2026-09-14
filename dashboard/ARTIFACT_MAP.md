# Dashboard source map

Every displayed analytical value comes from a frozen source. The dashboard does not
recalculate model outputs or load model binaries.

| Dashboard section | Frozen source artifact(s) | Analytical scope |
|---|---|---|
| Thesis overview and split design | `config/analysis_contract.yaml`; `artifacts/post_model/tables/dataset_summary.csv`; production run/selection manifests | Frozen study design and governance |
| Class and placement balance | `artifacts/post_model/eda/class_distribution.csv`; `artifacts/post_model/eda/placement_distribution.csv` | Full frozen dataset, descriptive |
| Top-four vs bottom-four summaries | `artifacts/post_model/eda/composition_structure_by_outcome.csv` | Full frozen dataset, approved descriptive aggregates |
| F1–F5 progression | `artifacts/thesis_extension_v1/tables/feature_family_validation_comparison.csv`; `feature_family_definitions.csv` | Development validation; fixed logistic family comparison |
| Candidate model comparison | `artifacts/post_model/tables/validation_model_summary.csv`; `cv_model_summary.csv` | Development validation/CV only |
| Locked primary result | `artifacts/post_model/tables/final_test_model_summary.csv`; `final_test_metric_ci.csv` | Single frozen final test |
| Confusion matrix, ROC and PR curves | `final_test_confusion_matrix.csv`; `final_test_roc_points.csv`; `final_test_pr_points.csv` | Frozen final-test outputs; no prediction regeneration |
| Global interpretation | `artifacts/thesis_extension_v1/tables/shap_feature_importance.csv` | Approved frozen TreeSHAP validation analysis |
| Grouped interpretation | `artifacts/post_model/tables/feature_group_importance.csv` | Frozen permutation importance grouped by documented prefixes |
| Exact placement | `artifacts/thesis_extension_v1/tables/placement_final_test_metrics.csv`; `placement_match_bootstrap_ci.csv`; `placement_final_test_confusion_matrix.csv` | Secondary, single frozen final test |
| Reproducibility | `verification_report.json`; `post_model_manifest.json`; production run and selection manifests; `dashboard/dashboard_manifest.json` | Recorded provenance plus live read-only hash verification |
| Rank-stratified context | `artifacts/post_model/tables/test_metrics_by_rank.csv` | Optional descriptive final-test analysis; excluded from primary feature matrix |
| Top-four composition analysis | `artifacts/post_model/composition_analysis_v1/tables/*.csv` | Read-only descriptive associations, train-only adjusted regression, and validation TreeSHAP; identifier-free presentation tables |
| Composition evidence | `artifacts/post_model/evidence_synthesis/feature_evidence_table.csv`; `raw_vs_adjusted_associations.csv`; `interaction_strength_comparison.csv` | Integrated descriptive, adjusted and predictive evidence; null where no defensible mapping exists |
| Robustness and limitations | `support_threshold_sensitivity.csv`; `threshold_rank_stability.csv`; `multiple_testing_summary.csv`; `train_validation_descriptive_stability.csv`; `artifacts/post_model/limitations_matrix.md` | Predeclared sensitivity and development-only replication; no final-test discovery |

The composition page uses an explicit loader allow-list. Its filters change only
the presentation of generated aggregate tables; dashboard code never loads participant
rows, model binaries, or prohibited identifiers.

## Deliberately omitted

- Participant-level and match-level error explorers are omitted because the available
  error tables contain prohibited hashes/identifiers.
- No live scoring, recommendations, counterfactuals, model loading, prediction
  regeneration, SHAP recomputation, bootstrap recomputation, or threshold controls are
  implemented.
- Raw rank snapshots are not loaded and are never incorporated into the primary model.
- The dashboard does not attempt to repair missing artifacts. Missing optional output is
  reported as unavailable.

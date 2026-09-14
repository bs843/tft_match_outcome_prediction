# Proposal-to-final mapping

| Original objective | Final analysis | Artifact | Status | Deviation / reason |
|---|---|---|---|---|
| Compare top-four and bottom-four composition patterns | Support-filtered unit/item/trait/star/interaction analysis with Wilson intervals, odds ratios and BH correction | `composition_analysis_v1/tables/` | Completed and strengthened | Added inference and robustness |
| Evaluate increasingly detailed feature sets | Frozen cumulative F1–F5 validation comparison | `evidence_synthesis/feature_family_incremental_value.csv` | Completed | None |
| Interpret the selected model | Locked XGBoost validation TreeSHAP plus association comparison | `evidence_synthesis/shap_vs_descriptive.csv` | Completed and strengthened | No model refit |
| Predict exact placement | Locked random-forest secondary evaluation | `thesis_extension_v1/tables/placement_final_test_metrics.csv` | Completed | Secondary task |
| Examine rank | Retrieval-time rank contextual descriptions only | `composition_analysis_v1/tables/rank_stratified_feature_analysis.csv` | Partial / contextual only | Rank is not historical match-time rank and remains outside prediction |
| Analyse augments where available | No augment analysis | `composition_analysis_v1/tables/augment_top4_analysis.csv` | Not analysed | Absent from frozen participant schema; not inferred or externally merged |

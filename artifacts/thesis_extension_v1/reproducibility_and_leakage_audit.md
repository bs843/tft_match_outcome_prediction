# Reproducibility and leakage audit

- Master seed: 4275; post-hoc stability seeds: 4275, 4276, 4277.
- Outer partition: frozen chronological 3,500/750/750 match split; canonical hash recorded in the manifest.
- CV: five-fold `StratifiedGroupKFold`, grouped by match. Every match's eight participants remain together.
- Fold preprocessing: `CompositionFeatureEncoder` is newly fitted on each CV training fold. Vocabulary frequency filtering is therefore fold-local. Held-out folds are transform-only; unseen categories create no columns.
- Missing values: deterministic numeric zero fill inside the encoder. Scaling: none. Resampling: none. Hyperparameter search: none.
- Ablations: fixed XGBoost contract parameters; only the prespecified substantive feature groups change.
- Placement CV: fixed registered random forest and dummy baseline; no selection or tuning.
- TreeSHAP: native XGBoost `pred_contribs` on the frozen development XGBoost and validation matrix. Values are additive log-odds contributions.
- Probability threshold: frozen 0.5. No alternative threshold was selected.
- Final test: read-only descriptive metrics for models already locked before the original single test evaluation; no model, feature, or threshold decision uses these results.

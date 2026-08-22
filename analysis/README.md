# Offline thesis analysis pipeline

This package is downstream of collection and post-collection validation. It does
not call Riot endpoints, read raw match JSON, change the accepted cohort, or
produce live-game predictions.

## Production use

Install the pinned environment, then run:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m analysis.run --validation-run ACTUAL_RUN_ID
```

Use the literal run ID, without angle brackets. The command stops before creating
an analytical dataset unless the collection manifest and validation artifacts
prove all of the following:

- the run completed with the frozen 5,000-match target;
- the accepted-match manifest has exactly 5,000 unique entries and its checksum
  matches the collection manifest;
- offline validation is `PASS` or `PASS_WITH_WARNINGS`, with no critical, schema,
  integrity, participant, end-result, or Patch 17.8 failures;
- all 5,000 `validated_manifest.csv` entries are valid and resolve inside the
  run-specific sanitized directory;
- dataset, schema, configuration, and collection provenance match the frozen study.

Production analysis also requires the `analysis` package and dependency files to
be committed, so the exact executing code has an immutable Git SHA.

The analysis reads only sanitized JSON named by
`data/validated/<run-id>/validated_manifest.csv`. Private raw JSON is never an
analysis input.

## Engineering dry run

```powershell
python -m analysis.run --dry-run
```

This uses generated records and labels every run
`NON-ANALYTICAL TEST/DEVELOPMENT ONLY`. It bypasses the production gate through a
separate code path and cannot authorise a failed or undersized real collection.

## Outputs

Each invocation creates `outputs/<analysis-run-id>/` containing the participant
dataset, persisted match-grouped 80/20 split, F1–F5 matrices, EDA tables and SVG
figures, cross-validation and untouched-test metrics, fitted models, provenance,
and artifact checksums. Large tabular datasets/matrices use gzip-compressed CSV;
feature vocabularies are learned on development data only.

## Rank and leakage policy

Clean `rank_tier`, `rank_division`, and `rank_status` are retained only for
description and stratification. Rank snapshot timestamps, queue type, LP,
wins/losses, request bookkeeping, identifiers, targets, and post-outcome fields
are prohibited from primary features. `analysis/configs/columns.yaml` is the
machine-readable contract; violations abort execution.

`platform_region` and `regional_route` are likewise retained as audit metadata,
never primary features. Regional participant, placement and rank distributions
and a match-level subgroup manifest are emitted for later robustness evaluation.
No held-out-region model is fitted automatically.

Primary modelling predicts `top4` derived from placement. Exact placement is a
secondary task. Results are descriptive predictive associations, not causal
effects. Rank-enriched sensitivity modelling requires a separately labelled,
pre-specified amendment and is not enabled here.

Primary Top-4 baselines comprise DummyClassifier, Logistic Regression, Random
Forest, XGBoost 3.0.2, LightGBM 4.6.0, and CatBoost 1.2.8. The three boosting
libraries are mandatory for production analysis. A dry run records a clearly
marked `dependency_unavailable` row for any booster not installed in the current
development environment.

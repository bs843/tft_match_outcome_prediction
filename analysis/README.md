# Offline thesis analysis pipeline

This package is downstream of collection and post-collection validation. It does
not call Riot endpoints, read raw match JSON, change the accepted cohort, or
produce live-game predictions.

## Production use

Install the pinned environment, then run:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m analysis.preflight
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

Production analysis requires contract `1.0.3`, a clean worktree, and HEAD exactly
at annotated tag `tft-patch17.8-analysis-contract-v4`. The contract binds the
collection/dataset manifests, authoritative split, scientific configuration,
column roles, model and metric registries, seeds and test-access policy by hashes.

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

Each production invocation consumes the immutable chronological 70/15/15 split:
3,500 train, 750 validation and 750 test matches. No split is regenerated.
Feature vocabularies and preprocessing are fitted on train only. Validation alone
selects each task's model. Independently locked top-four and exact-placement
selection manifests must pass integrity checks before test is opened once for
final evaluation. Every task access is recorded in `run_manifest.json` before the
test payload is read.

The old deterministic 80/20 helper is deprecated engineering-only behaviour. It
is unreachable from the production entry point and can never produce thesis-valid
artifacts.

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

## Frozen Analysis Protocol

Dataset freeze and analysis freeze are separate milestones. The immutable dataset
is `tft_set17_patch17_8_v11`, sourced from collection tag
`tft-patch17.8-final-collection-v10`. The analysis protocol is defined by
`config/analysis_contract.yaml` and released as
`tft-patch17.8-analysis-contract-v4` only after the complete test suite and
candidate preflight pass. Configuration changes require a new contract version
and tag; production has no force or ignore-contract bypass.

Contract `1.0.2` superseded but did not move the immutable v1/v2 tags. It made the
CI tests self-contained because private/ignored collection artifacts are not
present in source checkouts; real-data verification remains the responsibility of
the production preflight on the authorized analysis host. Text configuration
hashes normalize LF/CRLF so the same approved YAML verifies on Linux and Windows.

Contract `1.0.3` is the minimal exact-placement governance amendment. Version
1.0.2/v3 remains immutable historical evidence but is not authorized for
production modelling because it allowed both placement candidates to reach test
without validation selection. Under v1.0.3, dummy and random forest are fitted on
train and compared on validation MAE; lowest MAE wins, with registry order
(`dummy`, then `random_forest`) resolving exact ties. The placement selection is
locked and integrity-checked before test access, and only its winner is evaluated
on test. The test partition is not used to compare exact-placement candidate
models. Dataset, split, features, seed, preprocessing and top-four behaviour are
unchanged.

Rank-snapshot variables remain descriptive metadata only and are rejected from
all primary matrices. Train is used for fitting, validation for selection, and
test only for one final locked evaluation. Contract mismatches are always NO-GO.

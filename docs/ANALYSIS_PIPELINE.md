# Dissertation analysis protocol

## Analytical unit and targets

The analytical unit is a participant's final board. Every approved match supplies
eight rows with unique placements 1 through 8. `top4` is defined as
`placement <= 4`; exact placement is the secondary eight-class target. Both are
excluded from `X`. The analysis models associations between final-board
composition and observed outcome; it is not live prediction and makes no causal
claim.

Platform region is carried from the sanitized validation record for provenance,
subgroup summaries and optional future held-out-region evaluation. It is excluded
from every primary feature family.

## Feature families

| Family | Contents |
|---|---|
| F1 | level and board size |
| F2 | F1 plus unit presence and star level |
| F3 | F2 plus trait presence and activation strength |
| F4 | F3 plus item counts |
| F5 | F4 plus board-structure summaries |

Vocabularies are fitted only on the development partition. Categories below the
configured frequency are omitted; unseen test categories do not create columns.

## Splitting and evaluation

Production consumes the existing chronological match-group split: 3,500 train,
750 validation and 750 test matches. It never generates assignments and fails if
the frozen manifest/hash/counts/cohort differ. Seed 4275 governs model randomness
and five-fold grouped diagnostics within train. Validation alone selects the
primary model. Test remains unopened until selection is locked, then is evaluated
once. No match can cross partitions or folds.

Top-four metrics are accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and the
confusion matrix. Exact-placement metrics are MAE, accuracy, macro F1, and
weighted F1. Primary baselines are dummy, logistic regression, random forest,
XGBoost 3.0.2, LightGBM 4.6.0, and CatBoost 1.2.8. Dummy and random forest provide
secondary placement baselines. Boosting parameters are intentionally conservative;
this stage is baseline comparison rather than extensive hyperparameter optimisation.

## Provenance and eligibility

Each run records collection, dataset/schema/tag/commit, analysis contract/tag,
input/configuration and
artifact checksums, runtime, split, and reproduction command. Production analysis
is fail-closed: missing evidence is a failure, not an assumption. Failed or
capacity-failed collections and the 100-match pilot are ineligible for thesis
results. The pilot may be used only for engineering outside the production entry
point.

## Frozen Analysis Protocol

Contract `1.0.0` is machine-readable at `config/analysis_contract.yaml`. It binds
dataset `tft_set17_patch17_8_v11`, collection tag
`tft-patch17.8-final-collection-v10`, the 70/15/15 split, configuration hashes,
features, models, hyperparameters, targets, metrics, seeds, artefacts and test
policy. The analysis release tag is `tft-patch17.8-analysis-contract-v1`.
Production preflight requires a clean checkout at the exact annotated tag commit.
Any scientific change requires a new contract version and immutable release tag.

The earlier 80/20 split is deprecated and non-production. Rank-snapshot fields,
identifiers, targets, provenance, timestamps and progression outcomes are excluded
from primary X matrices. There is no production bypass for a contract mismatch.

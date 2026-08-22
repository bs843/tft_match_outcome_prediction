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

Seed 4275 produces a deterministic 80/20 split over sorted unique match hashes.
Five-fold stratified group cross-validation occurs within development data. No
match can cross partitions or folds.

Top-four metrics are accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and the
confusion matrix. Exact-placement metrics are MAE, accuracy, macro F1, and
weighted F1. Primary baselines are dummy, logistic regression, random forest,
XGBoost 3.0.2, LightGBM 4.6.0, and CatBoost 1.2.8. Dummy and random forest provide
secondary placement baselines. Boosting parameters are intentionally conservative;
this stage is baseline comparison rather than extensive hyperparameter optimisation.

## Provenance and eligibility

Each run records collection, dataset/schema/tag/commit, input/configuration and
artifact checksums, runtime, split, and reproduction command. Production analysis
is fail-closed: missing evidence is a failure, not an assumption. Failed or
capacity-failed collections and the 100-match pilot are ineligible for thesis
results. The pilot may be used only for engineering outside the production entry
point.

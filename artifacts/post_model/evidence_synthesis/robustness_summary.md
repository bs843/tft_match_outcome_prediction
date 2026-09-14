# Robustness summary

- Support thresholds were predeclared: 100/200/300/500 for units, items, traits and unit-star rows; 100/150/250/500 for exploratory interactions.
- Rank correlations and top-10/top-20 overlaps are in `threshold_rank_stability.csv`.
- Multiple-testing counts are in `multiple_testing_summary.csv`; BH q-values are the sole multiplicity correction.
- Train-versus-validation descriptive stability uses only development partitions. No final-test outcome was used for discovery or threshold choice.
- 159/182 supported development comparisons retained direction.
- Rank remains retrieval-time context only and is absent from the frozen primary model.

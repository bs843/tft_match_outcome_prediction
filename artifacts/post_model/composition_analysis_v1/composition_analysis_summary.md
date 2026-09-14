# Composition analysis summary

## Frozen-cohort validation

- 40,000 participant end-state observations from 5,000 complete matches.
- Top-four baseline: 50.0%; average placement: 4.500.
- Every match remains wholly within one frozen split. No participant or match hashes are exported.
- Available feature types: units, unit star levels, equipped items, traits with observed activation tier and contributing-unit count, retrieval-time rank tier, player level, and composition-structure summaries.
- Augments are absent from the frozen participant schema and were not analysed.

## Main descriptive findings

Among units with at least 200 appearances, **Shen** had the largest positive uplift
(65.8%; +15.8 pp), while
**Aurora** had the strongest negative association
(38.8%; -11.2 pp).
The strongest supported item was **Force Of Nature** (77.0%;
+27.0 pp). The strongest supported observed trait signature was
**Space Groove · tier 4 · 7 units** (70.8%;
+20.8 pp).

The highest-uplift supported exploratory interaction was **Hptank · tier 1 · 2 units +
Shen Unique · tier 1 · 1 units** (70.8%, n=2749).
Interaction rankings are exploratory and are reported only above 150 observations.

## Model interpretation and feature families

The largest mean-absolute TreeSHAP feature was **Structure · Total Items**. TreeSHAP explains the
frozen development XGBoost's validation predictions; it does not estimate causal effectiveness.
Cumulative frozen logistic validation F1 increased most when items were added to F3
(+0.035);
F5 added only +0.002 over F4.

## Inference and limitations

Wilson intervals, exposure odds ratios and Benjamini–Hochberg q-values accompany large-scale
descriptive comparisons. The adjusted regression uses the frozen training partition only and
match-cluster-robust standard errors. These remain observational end-state associations.
Expensive units, high star levels, late-game items and high player level are subject to survivorship
and endogeneity. Rank is a retrieval-time snapshot, not historical match-time rank. Patch-specific
balance and rank subgroup sizes limit generalisation.

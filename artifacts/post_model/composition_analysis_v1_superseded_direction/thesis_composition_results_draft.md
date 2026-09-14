# 4.X Composition features associated with top-four placement

## 4.X.1 Unit-level patterns

The frozen cohort contained 40,000 participant boards and had an exact top-four rate
of 50%. Among units satisfying the prespecified 200-appearance threshold,
Shen showed the largest positive top-four uplift (+15.8
percentage points), whereas Aurora showed the largest negative uplift
(-11.2 percentage points). Estimates for less common units remain
in the machine-readable table but are flagged and excluded from supported rankings.

## 4.X.2 Item-level patterns

At board level, Force Of Nature was most positively associated with top-four placement among
items meeting the 200-board threshold. Holder-conditional unit–item results additionally
compare each combination with the unit's and item's standalone rate, reducing—but not removing—the
confounding created when items are concentrated on stronger holders.

## 4.X.3 Trait-level patterns

Traits were evaluated using observed activation signatures combining `tier_current` and the number
of contributing units. Space Groove · tier 4 · 7 units had the highest supported uplift. These are
observed payload signatures rather than inferred canonical breakpoint labels.

## 4.X.4 Star-level and composition interactions

Unit–star estimates showed systematically different end-state rates across upgrade levels for many
well-supported units. Limited unit–item, trait–unit and trait–trait combinations were retained only
when support reached 150. The highest supported interaction was
Hptank · tier 1 · 2 units with Shen Unique · tier 1 · 1 units; all interaction results are exploratory.

## 4.X.5 Rank-tier differences

Rank-stratified estimates were retained for the project's observed tiers and flagged below
100 observations per cell. Variation across tiers describes heterogeneous association
patterns and should not be interpreted as evidence that rank modifies a causal treatment effect.

## 4.X.6 Adjusted feature associations

The adjusted training-partition logistic model included standardized basic/structure controls,
retrieval-time rank indicators, and a limited set of common units, traits and items. Odds ratios use
match-cluster-robust confidence intervals. Adjustment did not convert the observational end-state
design into a causal analysis; survivorship and endogenous board formation remain central limitations.

## 4.X.7 Model-based feature importance

Native TreeSHAP values from the frozen development XGBoost were calculated on validation data.
Structure · Total Items had the largest mean absolute attribution. Positive and negative SHAP
values describe contributions to the model's top-four log odds, not effects on match outcomes.

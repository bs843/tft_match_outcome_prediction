# Extended thesis analysis results

## Scope and safeguards

This is a read-only, post-selection extension of frozen analysis `analysis-20260825T190121Z-1b3eaaff` under
contract `1.0.4`. It does not change the accepted cohort, frozen chronological
split, registered models, selected models, thresholds, or final-test decision. Ablation,
TreeSHAP, correlation, repeated-CV, and placement-CV analyses use training or validation data;
the final test is used only for descriptive evaluation of the already locked models.

## Feature families

F1–F5 are cumulative. F1 contains basic final-board state; F2 adds units and star levels;
F3 adds traits; F4 adds items; and F5 adds structural summaries. The exact operational
definitions and undocumented semantic details are in `tables/substantive_unit_definitions.md`.

## Strongest predictive unit

Full F5 XGBoost fold-local CV F1 was 0.848 (SD 0.005).
By the prespecified ablation definition, **Composition structure**
was strongest: removing it produced the largest mean deterioration
(0.0246 F1). This is predictive importance, not causal influence.
TreeSHAP's largest aggregate unit was **Composition structure**.
Differences between ablation and TreeSHAP are expected because correlated or redundant columns
can share attribution; 137 feature pairs had absolute validation correlation ≥0.90.

## Model selection and stability

LightGBM had the slightly higher mean CV F1 (0.8500 versus 0.8494),
but the frozen selection rule maximized validation F1, where XGBoost scored
0.8543 versus 0.8525. The paired-fold comparison does
not justify claiming either leading model is intrinsically superior. Logistic regression remains
the parsimonious benchmark. Across 15 post-hoc grouped folds over three deterministic seeds,
XGBoost F1 averaged 0.848 (fold SD 0.004).

## Exact placement

Fold-local random-forest placement CV achieved accuracy 0.333, macro F1
0.323, weighted F1 0.323, and MAE
1.118 (fold SD 0.010), compared with dummy MAE
3.500. The locked final-test random forest achieved MAE
1.065; exact-placement prediction is meaningfully better than chance but moderate.

## Probability evaluation

At the frozen threshold 0.5, test Brier score was 0.106, log loss
0.336, and 10-bin ECE 0.009.
No application-specific alternative threshold is documented, so none was invented or optimized.

## Dependence and limitations

Matches never cross partitions or CV folds, and the outer split is chronological. Participant
overlap exists across 3 of three partition pairs, so match
grouping does not eliminate every player-level dependence. All data belong to one patch, set,
and queue; no cross-season generalization is established. TreeSHAP and ablations describe model
prediction, not causal effects. Small performance differences and correlated features require
cautious interpretation.

## Defensible conclusions

- Top-four classification is strong and generalizes well under the frozen split.
- Exact-placement prediction is better than the dummy baseline but remains moderate.
- F5 is the strongest combined frozen family; F4 is nearly equivalent in the frozen logistic family comparison.
- The new fold-local ablation—not the earlier aggregate comparison—identifies the strongest substantive predictive unit.
- Model importance is predictive and associational, not causal.

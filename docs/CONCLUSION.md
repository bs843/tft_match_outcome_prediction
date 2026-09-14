# Chapter 5: Conclusions

## 5.1 Introduction

This study examined whether final-board composition features could distinguish top-four from bottom-four finishes in ranked standard Teamfight Tactics matches. It used 5,000 completed Set 17 matches from Patch 17.8, representing 40,000 participant-level final boards. The results support strong retrospective classification within this population, but they do not establish causal effects or live-game predictive utility because every predictor was observed at the end of the match.

## 5.2 Conclusions for the research questions

### 5.2.1 Composition associations and adjustment

RQ1 asked which end-state composition features were associated with top-four placement. Supported units, items, traits, star levels, interactions, and structural summaries showed materially different outcome rates. Shen had the largest positive uplift among sufficiently supported units, Aurora had the largest negative uplift, Force Of Nature led the supported item comparisons, and the Space Groove tier-4, seven-unit signature led the supported trait configurations. These rankings describe the Patch 17.8 cohort; they do not show that selecting a feature would cause a better placement.

RQ2 asked whether selected associations persisted after adjustment for observed board-state and participant covariates. Several common-unit relationships, including those for Shen, Blitzcrank, and Morgana, retained positive adjusted associations, whereas some item estimates weakened. The adjusted odds ratio for Frozen Heart, for example, was close to one and its confidence interval included one. Adjustment therefore refined the descriptive evidence but could not account for unobserved economy, shop offers, positioning, opponent state, combat randomness, or the sequence of decisions that produced the final board.

### 5.2.2 Predictive contribution and model interpretation

RQ3 examined the incremental predictive information provided by cumulative feature families. Adding unit identities and star levels to basic board state produced the largest improvement in validation F1, followed by the addition of item information. Traits and structural summaries produced smaller subsequent gains because much of their information was already represented by earlier feature families. The comparison therefore indicates the order in which information became available to the cumulative model, rather than an isolated ranking of the substantive importance of each family.

RQ4 examined which feature groups contributed most strongly to the selected XGBoost model. Composition structure was the strongest substantive predictive unit: it performed best as a unit-only representation and produced the largest deterioration when removed from the complete feature set. TreeSHAP provided corroborating evidence, with item count, mean star level, player level, Thief's Gloves count, and three-star-unit count among the leading individual contributors. Ablation loss, SHAP contribution, adjusted association, and descriptive uplift measure different quantities and should not be collapsed into a causal importance ranking.

### 5.2.3 Exact placement and robustness

RQ5 asked to what extent end-state features could estimate exact placement. The selected random forest clearly exceeded the dummy baseline, achieving final-test accuracy of 0.3493 and mean absolute error of 1.0648 positions. Although exact classification was moderate, 72.35% of predictions were within one position and 89.92% were within two. Final-board composition therefore captured broad placement order but could not reliably distinguish neighbouring outcomes.

RQ6 examined robustness. The selected top-four classifier achieved final-test F1 of 0.8537 and ROC-AUC of 0.9305, closely matching validation performance. Repeated grouped cross-validation and the participant-disjoint sensitivity produced similar F1 estimates. Most supported development associations also retained their direction between training and validation. Robustness was strongest for aggregate predictive performance and more qualified for individual feature rankings; all checks remained internal to the same patch cohort.

## 5.3 Overall answer to the research problem

Final-board composition contains substantial information about whether a participant finished in the top four in the sampled Patch 17.8 matches. Unit identities and upgrades supplied the largest new information beyond basic board state, items supplied the next major increment, and compact structural summaries of board maturity formed the strongest predictive representation. The similarity of validation and test performance supports the conclusion that this pattern was reproducible within the frozen cohort rather than an isolated favourable split.

The timing of measurement defines the boundary of this conclusion. A final board records strategic choices, but it also records accumulated resources, survival, random opportunities, and prior success. The model may therefore recognise the consequences of a successful game as well as configurations associated with success. It should be understood as a retrospective outcome classifier, not as an early-game forecast, causal analysis, or coaching system.

## 5.4 Contribution of the study

Empirically, the study provides a large, multi-region account of final-board composition and placement within one defined TFT patch. It combines supported descriptive comparisons with adjusted estimates, incremental feature-family experiments, exact-placement analysis, and model-based interpretation.

Methodologically, the study distinguishes four forms of evidence that are often conflated in applied machine learning: observed prevalence differences, selected-covariate associations, incremental predictive contribution, and fitted-model attribution. Match-level grouping, train-only preprocessing, prespecified model selection, and match-cluster uncertainty intervals strengthened the evaluation and reduced common leakage risks.

Technically, the versioned collection and analysis workflow provides an auditable link between source records and reported aggregate results. Its principal research value lies in supporting the integrity and reproducibility of the empirical analysis; detailed system requirements, implementation traceability, and operational recovery procedures are retained in the repository rather than treated as separate scientific findings.

## 5.5 Limitations affecting the conclusions

The principal limitation is the observational end-state design. Survivorship and endogeneity are likely because stronger players and favourable earlier states improve access to levels, upgrades, items, and expensive units. Selected-covariate adjustment cannot recover the unobserved choices and opportunities needed for causal interpretation.

The findings are patch- and population-specific. Balance changes can alter unit, item, and trait relationships, and the regional analyses were internal subgroup checks rather than independent external validation. Retrieval-time rank was not a historical measure of skill, while augments were unavailable in the frozen participant schema. Neither was used as a primary predictor.

Match grouping controlled the strongest within-match dependence, but some players occurred in multiple matches. A post hoc participant-disjoint analysis produced similar performance, although its uneven connected-component folds limited precision. Exact placement also has a weaker confirmatory status than the primary top-four analysis: an earlier superseded workflow had not completed independent validation selection before candidate comparison. Those outputs were excluded, and the final protocol used a separate selection record, but the task should remain labelled secondary.

Finally, the structural 50% prevalence of top-four outcomes follows from sampling all eight participants in complete matches. Calibration and predictive values may differ for filtered populations, other patches, or earlier game states, and the fixed threshold was not optimised for a practical decision context.

## 5.6 Recommendations and future research

The most valuable extension is time-indexed data collected at consistent stages of a match. Such data should record the composition available at the prediction time together with economy, level, items, augments, available choices, opponent state, positioning, and subsequent placement. This would support genuine forecasting and help separate prior advantage from the effect of the observed board.

Future studies should test transportability across patches, sets, and regions using newly frozen cohorts and prospective holdouts. Historical rank should be measured at or before the match, and participant-disjoint evaluation should be planned before analysis where the connected-component structure permits it.

Exact-placement research could compare ordinal regression, ranking losses, and within-match ranking models with the present nominal classifier. Causal questions would require a different design with temporally defined decisions, stronger measurement of available alternatives, and defensible control of time-varying skill, economy, and survival.

The present model should not be deployed for live coaching or player evaluation. Any practical system would first require a defined prediction time, permitted inputs, error costs, prospective calibration, independent validation under patch drift, and safeguards against presenting retrospective associations as strategic instructions.

## 5.7 Final conclusion

Final-board composition provided a strong and stable retrospective signal of top-four placement in this Patch 17.8 cohort, while exact placement remained materially harder to classify. The study's contribution is not a claim that particular game choices cause success, but a reproducible account of how end-state composition, board maturity, and observed outcome relate under a leakage-controlled evaluation. Temporal and cross-patch studies are required before this evidence can support forecasting or practical recommendations.

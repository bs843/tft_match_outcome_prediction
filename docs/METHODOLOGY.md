# Chapter 3: Methodology

## 3.1 Study design

This study used a retrospective, observational, offline design to examine the relationship between final-board composition and placement in ranked Teamfight Tactics (TFT) matches. The primary analytical task was binary classification of whether a participant finished in the top four. Exact placement from first to eighth was evaluated as a secondary task. Descriptive, adjusted, predictive, and model-interpretation analyses were kept separate because they answer different questions and support different levels of inference.

The analytical unit was one participant's board at the end of a completed match. Each accepted match supplied eight participant observations. The predictors described the final board rather than an earlier game state, so the analysis measured retrospective outcome classification. It was not designed to estimate causal effects, forecast from a live decision point, or recommend in-game actions.

The final dataset and analysis were governed independently through versioned collection and analysis protocols. Before production analysis, these protocols fixed the cohort, data split, feature rules, candidate models, metrics, random seeds, model-selection rules, and final-test access policy. The corresponding configuration, release identifiers, and checksums are retained in the reproducibility record.

## 3.2 Study population and data source

The study population comprised completed standard ranked TFT Set 17 matches played during public Patch 17.8. Pilot responses established that the public patch appeared as client patch 16.15 in the Riot match payload. The public and client patch identifiers were stored separately to avoid treating their differing numbers as a mismatch.

Match and ladder records were collected from Riot Games application programming interfaces. The sampling frame covered GOLD, PLATINUM, EMERALD, DIAMOND, and MASTER players from five platform regions: ME1, EUW1, EUN1, NA1, and KR. These platforms were routed through the EUROPE, AMERICAS, or ASIA regional clusters as required by the endpoint. The same rank-stratum design was applied to every platform. The purpose of the multi-region design was to broaden candidate discovery; it did not impose equal accepted-match quotas by region.

Only the frozen collection window was eligible. Match-history timestamps were used to limit candidate discovery, but final patch membership was determined from each downloaded match response. Time alone was not accepted as proof of the patch.

## 3.3 Sampling and data acquisition

### 3.3.1 Pilot and capacity planning

A 100-accepted-match pilot was used for engineering validation and capacity planning. It assessed API routing, match acceptance, duplication, request volume, schema fields, and identifier handling. The pilot was not eligible for thesis results and could not enter the production analysis.

Earlier single-region attempts showed that the available candidate pool was insufficient for the planned sample. The final design therefore expanded discovery across five platform regions while retaining the same patch, queue, rank strata, validation rules, and target. The main collection required exactly 5,000 accepted matches. No other patch could be substituted if the target was difficult to reach.

### 3.3.2 Ladder and match discovery

For GOLD through DIAMOND, the collector requested three ladder pages for each division from IV through I and selected no more than 160 unique players from each populated page. MASTER used one apex page. Selection and interleaving were deterministic under random seed 4275. Player identifiers were deduplicated before match-history requests.

For each selected seed player, up to three match-history pages of 100 identifiers were requested within the configured UTC window. Candidate match identifiers were globally deduplicated across seeds and regions. The collector retained the platform on which a candidate was discovered as source provenance.

Match-history endpoints operate at regional-cluster level and can return matches associated with another platform in the same cluster. Download routing was therefore derived from the match identifier prefix rather than assumed from the seed player's platform. Candidates whose prefixes fell outside the five frozen platforms were counted and excluded. Platform candidate lists were interleaved before download so that one region could not exhaust the acceptance target before the others received an opportunity.

The system imposed a capacity gate before downloading match payloads. At least 25,000 globally unique in-window candidate identifiers had to be available. A shortfall ended the run with terminal status `capacity_failed` before download, acceptance, rank enrichment, or analytical output. The gate measured whether collection was feasible; it was not an inclusion criterion for an individual match.

### 3.3.3 API request handling

The client applied bounded timeouts and retries, exponential backoff, and both locally configured and server-provided rate limits. Credentials were supplied through the environment and were excluded from source code, configuration, and logs. Authentication and rate-limit failures produced explicit, auditable statuses rather than silent substitution or uncontrolled retries. Detailed request and recovery settings are documented with the collection software.

### 3.3.4 Immutable raw storage and resumption

Downloaded responses were written atomically as private JSON files and recorded with SHA-256 checksums. Existing raw data could be reused only after checksum verification and validation under the current configuration; an earlier acceptance decision was never inherited. Checkpointing preserved completed work for audit, while failed or partial collection attempts remained ineligible for the production cohort. This prevented the final dataset from being assembled across incompatible attempts.

## 3.4 Match eligibility and validation

Each downloaded match was subjected to explicit inclusion checks. An accepted record had to meet all of the following conditions:

1. public Patch 17.8 and client patch 16.15;
2. Set 17;
3. queue identifier 1100;
4. standard game type;
5. a timestamp inside the verified collection window;
6. the observed `info.endOfGameResult` value `GameComplete`;
7. exactly eight unique participants;
8. placements one through eight, each occurring exactly once;
9. identical participant sets in the metadata and information sections;
10. valid participant structures with non-empty unit lists; and
11. a structurally valid trait list.

The completion field, participant count, participant-set agreement, and placement permutation were tested independently. A structurally plausible eight-player record could not substitute for a missing or incorrect completion value. Rejected matches were retained privately with explicit reasons and were excluded from the accepted manifest.

The main run froze its accepted cohort only after reaching exactly 5,000 unique eligible matches. The manifest recorded the accepted-match content checksum, configuration and Git provenance, regional fields, raw checksums, and run status. The production cohort consisted of collection run `20260823T124729Z-673b6e6c` and yielded 40,000 participant observations.

### 3.4.1 Offline validation gate

A separate read-only stage revalidated the completed collection before normalisation or analysis. It checked raw-file integrity, schemas, patch and completion status, participant and placement invariants, duplicate keys, joins, privacy controls, and prohibited-feature rules without contacting Riot or modifying source records. A sanitized analytical dataset was created only when no critical finding existed; otherwise, the collection remained ineligible for thesis analysis.

## 3.5 Rank snapshots

After the accepted match cohort was fixed, the collector requested one ranked-entry snapshot for each unique participant using the participant's most recent observed platform. A failed lookup did not remove a participant or alter accepted cohort membership. The recorded tier and division represented rank at retrieval time rather than historical rank when the match was played. Rank variables were therefore excluded from every primary predictive feature matrix and used only in explicitly labelled contextual, adjusted, or subgroup analyses.

A routing failure required a downstream-only recovery after the match cohort had been frozen. Integrity checks confirmed that recovery could retry rank requests and rebuild dependent artifacts but could not add, remove, or replace matches. The original failure and recovery remain available in the technical audit record.

## 3.6 Privacy and data management

Raw Riot responses and identifier-bearing acquisition metadata were treated as private research data. API credentials and the pseudonymisation key existed only in the environment or ignored local `.env` file. Request logs contained endpoint templates, status codes, timing, attempts, and error classes rather than PUUIDs, Riot identifiers, credentials, or response bodies.

During normalisation, direct identifiers were replaced with project-specific HMAC-SHA256 pseudonyms generated with the secret `PARTICIPANT_HASH_KEY`. Match and participant pseudonyms supported joins and dependence audits without publishing Riot identifiers. These values remained pseudonymous rather than anonymous because they were stable within the project and originated from identifiable records.

Normalised data were separated into keyed match, participant, unit, unit-item, trait, and available-augment tables. Sanitized validation records removed display-name fields and source identifiers and rebuilt the top-four target from validated placement. Public reporting artifacts contained only aggregate values, schemas, configuration templates, code, checksums, figures, or sanitized fixtures. Private raw responses, match identifiers, PUUIDs, and reversible lookup material were excluded from publication and from the dashboard.

## 3.7 Analytical dataset and outcomes

### 3.7.1 Analytical unit

The model-facing dataset contained one row per participant per match, keyed internally by a pseudonymous match identifier and participant identifier. Every accepted match contributed exactly eight rows and one instance of each placement from one through eight. The participant's final units, their star levels and equipped items, active traits, and board-level summaries formed the source composition data.

### 3.7.2 Primary and secondary outcomes

The primary outcome, `top4`, was derived as one when validated placement was less than or equal to four and zero otherwise. Complete-match sampling produced four positive and four negative observations per match, creating an exactly balanced binary outcome without resampling or class weighting.

The secondary outcome was validated exact placement with labels one through eight. Placement was modelled as an eight-class target under the frozen analysis rather than converted into an ordinal regression problem. A grouped placement outcome was not enabled.

Placement, top-four status, and all placement-derived fields were excluded from the predictors. The outcome labels were created only after the source placement had passed the independent structural validation checks.

## 3.8 Feature engineering

### 3.8.1 Cumulative feature families

Five cumulative feature families were defined before production analysis. Each family retained the preceding groups so that its validation result measured the additional information associated with the newly added group. Table 3.1 gives their operational definitions.

**Table 3.1. Cumulative final-board feature families**

| Feature family | Included information | Operational variables |
| --- | --- | --- |
| F1 | Basic final-board state | Player level and number of units on the final board |
| F2 | F1 plus units | Unit-presence indicators and unit-specific star levels |
| F3 | F2 plus traits | Trait-presence indicators and recorded trait activation strength |
| F4 | F3 plus items | Count of each named equipped item |
| F5 | F4 plus composition structure | Unique units, mean and maximum star level, two- and three-star unit counts, total and unique items, active-trait count, and total recorded trait strength |

The F5 family was the primary feature specification. Augments were eligible where available, but the frozen participant schema contained no augment observations. They were not inferred or merged from another data source.

### 3.8.2 Encoding and missing values

The feature encoder was fitted once on the training partition for production analysis. Unit, trait, and item vocabularies retained categories appearing at least five times in training. Names were converted into deterministic column identifiers. Validation and test data were transformed with the frozen training vocabulary; an unseen category could not create a new column.

Unit presence was encoded as a binary value, and unit star level was encoded numerically with zero used when the named unit was absent. Trait presence was binary, while trait strength used `tier_current` and fell back to the payload's `style` field where necessary. Item features counted occurrences of each retained item. Structural variables were computed directly from the final-board collections. Missing numerical values were filled with zero only where zero represented the defined absence of a composition feature. No scaling, resampling, class weighting, or data-driven feature selection was applied in the production models.

### 3.8.3 Leakage exclusions

The feature contract prohibited direct and indirect outcome leakage. Placement, top-four status, grouped placement, last round, players eliminated, damage to players, and time eliminated could not enter the primary matrix. Participant and match identifiers, collection run values, timestamps, patch and queue fields, region provenance, validation fields, checksums, and source metadata were also excluded.

Every rank field was prohibited from the primary matrix because its retrieval occurred after the match and could reflect the studied result or later play. Column roles and prohibited prefixes were machine-readable. Feature construction and production preflight aborted when an unexpected or prohibited column appeared.

## 3.9 Data partitioning and dependence control

Matches were ordered chronologically and assigned to immutable training, validation, and test partitions in proportions of 70%, 15%, and 15%. The resulting split contained 3,500 training matches, 750 validation matches, and 750 test matches. All eight observations from a match remained in the same partition. The split manifest, counts, match membership, and canonical assignment hash were bound into the analysis contract. Production analysis consumed this manifest and could not generate a replacement split.

The training partition was used for fitting candidate models, fitting feature vocabularies, and five-fold cross-validation diagnostics. Validation data were used for model and feature-family comparison. The test partition was reserved for one final evaluation after selection. This chronological design provided a later held-out period rather than a random participant-level sample.

Five-fold `StratifiedGroupKFold` was used within training, with `match_hash` as the group and seed 4275. Grouping prevented a match from crossing the fitting and held-out portion of a fold. The feature encoder was fitted separately inside each training fold for the ablation, repeated-cross-validation, and placement diagnostics so that held-out categories did not influence the fold vocabulary.

Participant overlap was audited separately because the same player could appear in more than one match. The authoritative outer partitions contained no shared matches but did contain some repeated participants. A post hoc sensitivity grouped matches into connected components whenever they shared a participant and used `GroupKFold` to keep each component within one fold. This stricter analysis removed both match and participant overlap but did not replace the frozen split or change model selection.

## 3.10 Candidate models and model selection

### 3.10.1 Top-four classification

Six prespecified classifiers were compared on F5: a prior-probability dummy classifier, logistic regression, random forest, XGBoost 3.0.2, LightGBM 4.6.0, and CatBoost 1.2.8. Fixed settings were used rather than a hyperparameter search. Table 3.2 summarises the principal settings; seed 4275 was applied through each model's supported random-state parameter.

**Table 3.2. Candidate top-four classifiers and fixed settings**

| Model | Principal fixed settings |
| --- | --- |
| Dummy classifier | Prior strategy |
| Logistic regression | Maximum 2,000 iterations |
| Random forest | 300 trees, minimum leaf size 2, one processing thread |
| XGBoost | 300 trees, depth 6, learning rate 0.05, row and column subsampling 0.8, log-loss evaluation |
| LightGBM | 300 trees, 31 leaves, learning rate 0.05, row and column subsampling 0.8, deterministic column-wise training |
| CatBoost | 300 iterations, depth 6, learning rate 0.05, log-loss objective, one processing thread |

Each candidate received five-fold grouped-cross-validation diagnostics on the training-derived F5 matrix, then was fitted on the complete training partition and evaluated on validation. The primary selection rule chose the highest validation F1. Candidate registry order resolved an exact tie. Cross-validation means informed stability assessment but did not replace the frozen validation rule. The later ablation, repeated-cross-validation, and exact-placement diagnostic analyses fitted their vocabularies separately within each fold, as described in Sections 3.9 and 3.13.

The selected model name, parameters, validation metrics, feature-schema hash, tie-break rule, threshold, seed, and candidate results were written to an integrity-protected selection manifest. The binary classification threshold remained fixed at 0.5. Early stopping and threshold optimisation were disabled.

### 3.10.2 Feature-family comparison

A logistic-regression model with the same fixed implementation was fitted separately to F1 through F5 on training and evaluated on validation. This controlled comparison assessed the incremental predictive information of each cumulative family. It was not eligible to override the model selected from the F5 candidate registry.

### 3.10.3 Exact-placement analysis

The exact-placement registry contained a most-frequent dummy classifier and a random forest with 300 trees and minimum leaf size 2. Both candidates were fitted on F5 training data and evaluated on validation. The model with the lowest validation mean absolute error was selected, with registry order resolving an exact tie. The placement decision was stored in a separate selection manifest so that the primary and secondary tasks could not authorise each other.

### 3.10.4 Final-test isolation

The test partition could not be opened until both task-specific selection manifests existed and passed integrity verification. Test access was recorded in the run manifest before the test payload was read. The selected top-four and exact-placement models were then refitted on the combined training and validation partitions and evaluated once on test. Unselected models were not evaluated on test. Missing, modified, mismatched, early, or repeated access caused the production run to fail.

## 3.11 Evaluation measures and uncertainty

Top-four models were evaluated using accuracy, precision, recall, F1, receiver operating characteristic area under the curve (ROC-AUC), precision-recall area under the curve (PR-AUC), and a two-class confusion matrix. F1 was the primary selection metric because it summarised the balance between precision and recall at the fixed threshold. ROC-AUC and PR-AUC assessed ranking performance from predicted probabilities independently of a single classification threshold.

Post-selection probability evaluation added Brier score, log loss, and a reliability analysis with ten quantile-based bins. Expected calibration error summarised the weighted absolute difference between mean predicted probability and observed outcome frequency across the bins. No recalibration or alternative threshold was fitted on the final test.

Exact-placement evaluation used accuracy, macro F1, weighted F1, mean absolute error, median absolute error, and the proportions of predictions within one and two placement positions. Mean absolute error was the selection metric because it preserved the ordered distance between the predicted and observed placement even though the classifier produced one of eight class labels.

Uncertainty intervals were calculated with 2,000 deterministic percentile bootstrap samples using seed 4275. The resampling unit was the complete match rather than the participant row. Every selected match contributed all eight participant observations to a bootstrap sample, preserving within-match outcome dependence and the structural four-versus-four class balance. The 2.5th and 97.5th percentiles formed the reported 95% intervals.

## 3.12 Composition association analysis

### 3.12.1 Descriptive comparisons

The post-model composition analysis summarised the frozen cohort without changing model selection or final-test performance estimates. Binary exposure tables were constructed for units, board-level item possession, observed trait signatures, and unit-star combinations. Trait signatures combined the payload's current tier with the recorded number of contributing units; external canonical breakpoint labels were not inferred. Basic and structural board summaries were compared by outcome group. Component items and placeholder item identifiers were excluded from the ranked completed-item analysis.

For each exposed feature, the analysis reported support, pick rate, top-four count and rate, bottom-four count, mean and median placement, and percentage-point uplift from the cohort top-four rate. Ninety-five per cent Wilson intervals were calculated for the top-four proportion. Exposure odds ratios and log-scale confidence intervals compared exposed with unexposed participant boards. A 0.5 continuity correction was added when a two-by-two table contained a zero cell. Fisher's exact test provided the unadjusted p-value, and Benjamini-Hochberg q-values controlled the false discovery rate within the multiple feature comparisons.

The principal minimum support was 200 boards for units, items, and trait signatures; unit-star combinations required 100 observations. Results below a threshold remained available in machine-readable tables but were flagged and excluded from supported rankings. Limited unit-item, trait-unit, and trait-trait interactions required at least 150 observations and were labelled exploratory. Rank-stratified cells required at least 100 observations for supported interpretation.

The full-cohort descriptive rankings were conducted after the predictive model had been frozen and were not used to tune its features, parameters, model choice, or threshold. Separate train-versus-validation comparisons assessed directional stability without using test outcomes to choose a support threshold.

### 3.12.2 Holder and interaction analyses

Board-level item associations were supplemented by holder-conditional unit-item comparisons. These described the top-four rate for a unit-item combination relative to the standalone unit and item rates. Trait-unit and trait-trait analyses used the same participant-level outcome and minimum-support rules. Interaction excess was treated as exploratory descriptive evidence rather than biological or strategic synergy.

### 3.12.3 Adjusted association model

An unpenalised logistic-regression model was fitted on the frozen training partition for adjusted association analysis. Continuous context variables were standardised within training and included player level, composition size, mean star level, number of three-star units, total items, number of active traits, and total trait strength. Retrieval-time rank indicators were included as contextual covariates with GOLD as the reference when available. The ten most common units, items, and non-Unique trait signatures were encoded as binary terms.

Before fitting, one of any pair of terms with absolute correlation of at least 0.95 was removed deterministically to reduce near-exact redundancy. Odds ratios were obtained by exponentiating coefficients. Standard errors used a sandwich covariance estimator clustered by match, with finite-sample correction. Two-sided normal-approximation p-values, 95% confidence intervals, and Benjamini-Hochberg q-values were reported.

This model was intended to determine whether selected raw associations persisted after conditioning on recorded covariates. It was not used for primary prediction and did not support causal interpretation. Rank appeared only in this labelled contextual analysis and remained absent from the primary XGBoost matrix.

## 3.13 Model interpretation and sensitivity analyses

### 3.13.1 Unit-only and ablation analyses

The selected XGBoost specification was re-evaluated within fold-local grouped cross-validation to compare substantive feature groups. A full F5 model provided the reference. Unit-only variants retained one of basic state, units, traits, items, or structure. Leave-one-group-out variants removed one group at a time from F5. All variants used the same folds, seed, and fixed XGBoost parameters. The mean change in F1 from the paired full-F5 fold estimated the incremental deterioration associated with removal.

Unit-only performance and leave-one-group-out deterioration answer different questions. The former assesses how much outcome information one group carries by itself. The latter assesses information not recovered by the remaining groups. Both were reported because correlated and redundant encodings can make either quantity misleading in isolation.

### 3.13.2 TreeSHAP

TreeSHAP contributions were calculated from the frozen development XGBoost model on the validation matrix using XGBoost's native `pred_contribs` implementation. Values were expressed on the log-odds scale. Global feature importance was the mean absolute contribution across validation observations. Contributions were also summed and averaged within the substantive groups of basic state, units, traits, items, and composition structure.

SHAP analysis described how the fitted model allocated its predictions. It did not estimate how placement would change if a player intervened on a feature. Interpretation considered the high correlation among encoded variables because substitutes can share or transfer attribution.

### 3.13.3 Robustness analyses

Four groups of robustness checks were applied without reopening model selection. First, XGBoost grouped cross-validation was repeated with seeds 4275, 4276, and 4277. Second, player-connected-component folds removed participant overlap. Third, feature rankings were recalculated at support thresholds of 100, 200, 300, and 500, while the retained interaction artifacts permitted checks at 150, 250, and 500. Spearman rank correlation and top-list overlap compared ranking stability. Fourth, supported descriptive effects were calculated separately in training and validation to record directional agreement.

Multiple-testing summaries reported both raw p-values and Benjamini-Hochberg q-values. None of the robustness analyses changed the accepted cohort, selected model, feature family, classification threshold, or final-test decision.

## 3.14 Reproducibility and analytical governance

The software environment used Python 3.12 and a complete dependency lock file. Collection and analysis were tied to versioned configurations and source releases, while checksums linked the accepted cohort, data split, feature schema, model-selection decisions, and reported artifacts. Each analytical run recorded the inputs, configuration, random seeds, fitted parameters, selection result, test-access event, outputs, and reproduction command.

Post-model analyses verified their frozen inputs and wrote new versioned outputs without modifying the production run. Automated tests covered the principal collection, validation, privacy, grouping, leakage, selection, and artifact-integrity invariants using synthetic fixtures and mocked network responses. Exact versions, checksums, commands, and the full verification inventory are provided in the repository reproducibility documentation rather than repeated in the thesis narrative.

## 3.15 Ethical and legal considerations

The study used game records obtained through authorised Riot API access and did not involve intervention in player behaviour. Direct identifiers were required temporarily for collection and participant-level routing but remained in private storage. Public outputs used pseudonyms or aggregates, and the dashboard excluded participant-level error tables and identifier-bearing records.

Data minimisation was implemented by separating acquisition metadata from the analytical matrix and excluding identifiers, rank snapshots, timestamps, and provenance from primary predictors. Request logs omitted sensitive path values and response bodies. Access roles, encrypted backup, retention duration, and secure deletion dates were to be governed by the institutional data-management plan. Riot API terms and relevant TFT policies were required to be checked immediately before live collection.

The principal interpretive safeguard was to describe the findings as observational and predictive. The study did not evaluate individual players, infer intent, or claim that a feature caused success. Retrieval-time rank was not treated as a stable measure of skill at the match timestamp.

## 3.16 Protocol amendments

Protocol changes were versioned rather than applied silently. Pilot evidence led to deeper discovery and later expansion from a single region to ME1, EUW1, EUN1, NA1, and KR because the original candidate pool could not support the target cohort. These changes preserved the patch, rank strata, outcome, and match-validity rules. Download routing was also corrected to use the authoritative match-identifier prefix.

An earlier, superseded exact-placement workflow had compared candidates without first completing independent validation selection. Its outputs were treated as exploratory and were excluded from thesis-valid model comparison. The final protocol required a separate validation decision before recorded test evaluation, and exact placement remained a secondary analysis. Failed and superseded runs were retained only as audit evidence; implementation-only version changes are documented in the repository.

## 3.17 Chapter summary

The methodology combined a frozen multi-region collection with fail-closed validation, privacy-preserving normalisation, chronological match-group partitioning, train-only feature learning, prespecified model comparison, and independently locked final evaluation. The primary analysis compared six classifiers for top-four placement, while exact placement followed a separate two-model selection procedure. Descriptive and adjusted composition analyses, group ablations, TreeSHAP, calibration, match-cluster uncertainty intervals, and robustness checks extended interpretation without altering the selected models or final-test decision.

These methods were designed to answer a predictive research question while limiting leakage, repeated test use, identifier exposure, and unsupported causal interpretation. The resulting evidence applies to final-board observations from the frozen Patch 17.8 cohort and provides the basis for the results reported in Chapter 4.

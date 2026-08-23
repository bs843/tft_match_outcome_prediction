# Prespecified Methodology

## Design and estimand

This is an observational, offline, post-match classification study restricted to ranked standard TFT Set 17 matches from public TFT patch 17.8. Pilot responses establish that this public patch is encoded as client patch 16.15 in `game_version`; these identifiers are stored separately. The primary estimand is held-out discrimination and calibration of top-four finish from final-board composition. Results are predictive associations, not causal effects or evidence of live-game usefulness.

## Sampling

The sampling frame is constructed from fixed ladder pages for tier-by-division strata. PUUIDs are deduplicated before fixed-seed selection. Match histories are bounded by verified UTC patch times, paginated, and globally deduplicated. Patch membership is determined from the downloaded response, not inferred from time alone.

The 100-match pilot estimates duplication, rejection and request rates. The main candidate budget is calculated from these empirical rates. If 5,000 valid matches are unavailable, the achieved sample is reported; other patches are never substituted.

## Inclusion rules

An accepted response must have client patch 16.15 corresponding to public TFT patch 17.8, Set 17, an allowed ranked queue and game type, a timestamp in the study window, Riot's observed `info.endOfGameResult` value `GameComplete`, eight unique participants, placements 1–8 exactly once, matching metadata participant identifiers, and structurally valid units and traits. The participant and placement invariants remain independent structural checks rather than substitutes for the explicit completion field.

## Outcomes and features

The primary target is top-four finish. Secondary analyses may model ordinal placement or within-match ranking. The primary feature family contains units, star levels, equipped items, active traits and augments. Vocabularies and rare-category thresholds are learned using training matches only.

Placement-derived targets, progression fields, collection-time rank, identifiers, timestamps and provenance cannot enter the feature matrix. Level, gold and game length are reserved for a separately labelled sensitivity analysis.

After the accepted match cohort is fixed, one platform-routed ranked-entry request is
made per unique participant. The resulting rank is explicitly a retrieval-time snapshot,
not a historical rank at the match timestamp. Successful unranked responses, 404s and
exhausted request failures remain as explicit status rows rather than being dropped.
Because a retrieval-time rank can reflect the studied match and later play, every rank
snapshot field is excluded from the primary model; it may support descriptive subgroup
audits or a separately declared sensitivity analysis only.

## Splitting and evaluation

Matches are ordered by game timestamp and allocated 70/15/15 to training, validation and test. Every participant from a match remains with that match. Participant overlap is audited and disclosed. Strict player-connected-component splitting is a sensitivity analysis when feasible.

Models comprise a prior baseline and regularised logistic regression. Hyperparameters are selected on validation log loss after grouped cross-validation within training data. The selected pipeline is refit on training plus validation, then evaluated once on test.

Reported metrics are ROC AUC, log loss, Brier score, accuracy, F1 and calibration. Confidence intervals use bootstrap resampling of complete matches. The effective independent cluster count is the number of matches, not participant rows.

## Amendments

Any change to scope, sampling, validation, feature eligibility, split logic, models or metrics after inspecting test results must be documented as a protocol amendment and receive a new dataset or experiment version.

Dataset version `tft_set17_patch17_8_v3` increases the main-run allocation from
5 to 80 seed players per prespecified rank stratum. This pilot-informed change
uses the observed 58.69% match acceptance rate and approximately 9.68 unique
candidates per sampled player. Rank strata, eligibility rules, study scope,
schema and primary modelling design are unchanged.

Collection configuration `tft-patch17.8-final-collection-v3` and dataset version
`tft_set17_patch17_8_v4` increase discovery capacity after the v2 final attempt
exhausted 10,326 candidates with 3,142 accepted matches (30.43%). Non-apex ladder
pages increase from one to two and match-history depth from two to three pages.
The original tier/division strata, 5,000-match target, Patch 17.8 scope, eligibility
rules, schema 1.1.0 and modelling design remain unchanged. The 20,000-candidate
capacity target is a safety buffer, not an inclusion or validation threshold.

Collection configuration `tft-patch17.8-final-collection-v4` and dataset version
`tft_set17_patch17_8_v5` respond to the v3 audit: 31,818 candidate occurrences
collapsed to 13,585 unique IDs and 3,479 accepted matches. Non-apex ladder depth
therefore increases to three pages and deterministic selection increases to at most
160 unique players per populated page. History depth remains three pages. A main run
must establish at least 25,000 unique in-window candidate IDs before match downloads;
otherwise it terminates as a capacity failure. This gate concerns collection feasibility
only. Schema 1.1.0, rank strata, validity rules and primary modelling design are unchanged.
## Multi-region amendment

The v5 collection amendment retains Patch 17.8 and GOLD–MASTER while sampling the
ME1, EUW1, EUN1, NA1 and KR ladders. Each platform receives the same page and seed
opportunity, and candidate pools are interleaved before match evaluation. No equal
accepted-match quota is imposed. Candidate IDs are deduplicated globally before
the unchanged 25,000-capacity gate, and regional composition is reported.

The v6 routing amendment does not change the study population or sampling strata.
It recognizes that match history is retrieved at cluster level: the match-ID prefix
therefore determines candidate platform and download route, while the seed platform
remains discovery provenance. IDs from platforms outside the frozen five-platform
scope are reported and excluded before the capacity gate.

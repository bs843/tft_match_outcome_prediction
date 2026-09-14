# Technical Companion: Requirements Specification

## 1 Status and purpose

This document is retained as repository-level technical and audit documentation. It is not an examiner-facing thesis chapter unless the applicable programme rubric explicitly requires a software requirements specification. The examiner-facing account of the research design and analytical controls is provided in `METHODOLOGY.md`.

The specification covers the collection services, immutable data store, validation gate, analytical workflow, generated evidence, and read-only dashboard. Its requirements are derived from the frozen study configuration, the analysis contract, data-management rules, and implemented verification tests. They are stated as observable obligations for software maintenance, reproduction, and audit.

## 2 System scope

### 2.1 Product perspective

The system is a local, command-line research application with a separate Streamlit presentation layer. During acquisition, it communicates with Riot Games application programming interfaces to discover players and matches and to download completed match records. All later stages operate offline. The pipeline converts source responses into validated, pseudonymous relational data; constructs leakage-controlled features; trains and selects models; performs one governed final evaluation; and exports tables, figures, manifests, and a read-only dashboard.

The configuration and analysis contract form part of the system. They define the population, collection routes, validation rules, split assignments, eligible features, candidate models, metrics, random seeds, and test-access policy. Production runs must fail when the code, data, or configuration does not match these frozen definitions.

### 2.2 System boundary

The system includes the following components:

1. configuration and release-integrity checks;
2. multi-region player and match discovery;
3. rate-limited API access and resumable raw-data acquisition;
4. match validation and accepted-cohort freezing;
5. participant rank-snapshot collection for contextual analysis;
6. pseudonymisation, normalisation, and offline validation;
7. chronological grouped splitting and feature construction;
8. model training, selection, locked test evaluation, and uncertainty estimation;
9. evidence synthesis and artifact verification; and
10. read-only presentation of frozen results.

The system does not provide live-game predictions, player recommendations, automated strategic decisions, causal estimates, or a public store of raw Riot identifiers. It does not retrain models or regenerate statistical results from the dashboard. It is also not intended to generalise automatically to another patch, set, queue, or game mode.

## 3 Stakeholders and user classes

| Stakeholder | Interaction with the system | Requirement interest |
| --- | --- | --- |
| Researcher | Configures and runs collection, validation, analysis, and reporting commands | Correctness, recoverability, clear failures, and repeatable outputs |
| Supervisor or examiner | Reviews the study design, provenance, results, and limitations | Traceability, methodological compliance, and defensible interpretation |
| Reproducibility reviewer | Rebuilds results from frozen source data and manifests | Pinned dependencies, deterministic processing, checksums, and documented commands |
| Dashboard reader | Explores approved aggregate findings | Accurate labels, accessible navigation, and no exposure of identifiers |
| Data custodian | Controls private raw records, credentials, and pseudonymisation keys | Confidentiality, retention controls, and publication safety |
| Riot Games API | Supplies ladder, match-history, match, and ranked-entry responses | Correct routing, bounded request rates, authentication, and policy compliance |

The researcher is the only actor permitted to initiate collection or model execution. Dashboard readers receive access only to frozen, aggregate, identifier-free outputs.

## 4 Operating context and constraints

### 4.1 Study constraints

Production data are restricted to standard ranked TFT Set 17 matches from public Patch 17.8, represented by client patch 16.15 in the source response. A valid match must use queue identifier 1100, have the observed end result `GameComplete`, contain exactly eight unique participants, and contain each placement from one through eight exactly once. Match timestamps must fall within the verified patch window.

The frozen acquisition target is 5,000 accepted matches, producing 40,000 participant-level observations. The enabled platform regions are ME1, EUW1, EUN1, NA1, and KR, with their configured EUROPE, AMERICAS, and ASIA regional routes. Player discovery covers the GOLD through MASTER strata. These values may be changed only through a documented protocol amendment with new version and release identifiers.

### 4.2 Technical constraints

The software requires Python 3.12 and the pinned project dependencies. Live network access is required only for collection and rank-snapshot requests. Production analysis reads the sanitized validation layer and must not call Riot endpoints or read private raw responses. A production collection requires `RIOT_API_KEY` and `PARTICIPANT_HASH_KEY` values supplied through the environment rather than committed configuration.

### 4.3 Scientific constraints

The analytical unit is a participant's final board. All observations from a match must remain together during partitioning and grouped cross-validation. Feature vocabularies and preprocessing must be fitted without final-test information. Retrieval-time rank, identifiers, targets, outcome-progression fields, timestamps, and region provenance are not eligible predictors. The final test partition may be opened only after the primary and secondary model-selection decisions have been independently locked.

## 5 Principal use cases

### 5.1 Conduct a governed collection run

The researcher verifies the frozen Git state, supplies the required secrets, and starts either a pilot or main collection. The system samples configured ladder strata, discovers and deduplicates candidate matches, checks the capacity gate, downloads candidate records, validates each response, and stops when the target is reached or the candidate pool is exhausted. The successful postcondition is an immutable accepted-match cohort with a manifest, checksums, provenance, rejection counts, and explicit terminal status.

### 5.2 Validate and prepare the research dataset

The researcher runs the offline validation command for a specific completed collection. The system checks source hashes, schemas, match invariants, joins, privacy rules, and prohibited-feature controls. A sanitized validation dataset is created only when no critical error is present. Normalisation, splitting, and feature construction then produce pseudonymous tables, an immutable chronological split, and a train-derived feature vocabulary.

### 5.3 Run the frozen analysis

The researcher executes preflight checks and starts analysis with an approved validation run identifier. The system verifies the dataset, manifests, configuration hashes, release tag, split, feature contract, and clean working state. Candidate models are fitted on the training partition and compared on validation data. The chosen models are locked before a single final-test evaluation. The completed run records metrics, uncertainty intervals, model artifacts, selection decisions, test access, and provenance.

### 5.4 Review approved findings

The reader launches the dashboard and navigates the study overview, feature-family comparison, model performance, interpretation, exact-placement results, reproducibility evidence, composition analysis, and limitations. The dashboard reads allow-listed artifacts and verifies their schemas. It cannot load model binaries, participant-level error records, raw identifiers, or training libraries. Missing optional artifacts produce a visible unavailable state without changing frozen evidence.

### 5.5 Recover an authorised downstream failure

The researcher may invoke the versioned rank-snapshot recovery path only for the explicitly authorised run. The system first proves that the accepted cohort and raw checksums remain unchanged. Recovery may retry failed rank requests and rebuild downstream artifacts, but it cannot discover players, discover candidates, download new matches, or alter accepted membership. A separate recovery manifest preserves the original failure and the actions performed.

## 6 Functional requirements

### 6.1 Configuration and provenance requirements

| ID | Requirement | Acceptance condition |
| --- | --- | --- |
| FR01 | The system shall load study scope, platform routing, sampling, API, privacy, validation, feature, split, model, and path settings from version-controlled configuration. | Startup rejects missing, empty, inconsistent, or misspelled required values. |
| FR02 | A main collection shall run only from the exact commit referenced by the required annotated collection tag and with no tracked modifications. | A missing tag, a changed tracked file, or a different `HEAD` causes a pre-collection failure. |
| FR03 | Production analysis shall run only when the dataset, collection provenance, analysis contract, configuration hashes, split hash, and annotated analysis tag match the frozen contract. | Preflight reports no-go and creates no thesis-valid run when any bound value differs. |
| FR04 | Every run shall have an identifier and a manifest containing its status, timestamps, configuration version, dataset and schema versions, source commit, required tag, counts, and checksums. | The manifest is present and reconciles with the generated cohort and artifacts. |
| FR05 | Scientific changes shall require a new version and release record rather than modifying an earlier frozen contract. | Existing release identifiers remain immutable and a changed bound file fails verification. |

### 6.2 Acquisition requirements

| ID | Requirement | Acceptance condition |
| --- | --- | --- |
| FR06 | The collector shall sample the configured GOLD through MASTER ladder strata on every enabled platform using the fixed seed and page limits. | Sampling reports show each requested platform, tier, division, page, returned count, and deterministic selection count. |
| FR07 | The collector shall retain source platform as discovery provenance while deriving match download routing from the match identifier prefix. | A cross-platform history result is routed by its authoritative prefix, and an out-of-scope prefix is counted and excluded. |
| FR08 | Player identifiers and candidate match identifiers shall be globally deduplicated before the relevant downstream request. | Duplicate counts are reported and each unique target is requested at most once per applicable stage. |
| FR09 | A main run shall establish at least 25,000 unique in-window candidate match identifiers before any match download, acceptance decision, rank enrichment, or analytical output. | A shortfall ends the run with `capacity_failed` and none of the prohibited downstream stages executes. |
| FR10 | The API client shall enforce configured timeouts, retry limits, minimum intervals, fallback request windows, server rate-limit headers, and global `Retry-After` delays. | Simulated rate-limit responses delay and retry without exceeding the bounded policy. |
| FR11 | Confirmed credential expiry shall pause the pending request and check for a changed environment key for no longer than the configured recovery window. | A changed key resumes the same request; expiry of the recovery window produces a controlled failure. |
| FR12 | A participant-specific forbidden rank response shall be distinguished from general credential failure by a non-private credential-health probe. | A healthy probe records `request_failed`; an unhealthy probe enters credential recovery. |
| FR13 | Downloaded raw responses shall be written atomically, treated as immutable, and recorded with SHA-256 checksums. | Resume reuses a raw file only after its checksum matches provenance and its contents pass current validation. |
| FR14 | The system shall checkpoint completed work and preserve a terminal audit status when interrupted. | Restart does not repeat verified requests, and an interrupt records `partial` without presenting the run as complete. |
| FR15 | Previously failed, partial, or capacity-failed main runs shall not be resumed, merged, or silently promoted to a valid cohort. | Supplying a prohibited run identifier causes a fail-closed error. |

### 6.3 Validation and data-preparation requirements

| ID | Requirement | Acceptance condition |
| --- | --- | --- |
| FR16 | The match validator shall reject records outside Patch 17.8 and client patch 16.15, Set 17, queue 1100, standard game type, `GameComplete`, and the verified UTC collection window. | Every rejection has an explicit reason and no rejected record enters the accepted manifest. |
| FR17 | The match validator shall require eight unique participants, matching metadata and information participant sets, placements one through eight exactly once, non-empty units, and a traits list. | Any missing, duplicate, inconsistent, or malformed required structure causes rejection or a critical validation finding. |
| FR18 | The accepted cohort shall be frozen only when the main-run target of exactly 5,000 accepted, unique matches is reached. | The manifest count, accepted file, unique identifier set, and content checksum reconcile. |
| FR19 | The system shall create one rank-snapshot status row for every unique accepted participant, using `found`, `unranked`, `not_found`, or `request_failed`. | There are no missing or duplicate participant rows, and failed rank retrieval does not alter the accepted match cohort. |
| FR20 | Rank routing shall use the platform of each participant's most recent accepted match, with the greatest match identifier as the deterministic tie-break for equal timestamps. | Repeated execution on the same frozen inputs produces the same routing table and one logical snapshot per participant. |
| FR21 | Normalisation shall replace source identifiers with project-specific HMAC-SHA256 pseudonyms and shall separate matches, participants, units, items, traits, and available augments into documented keyed tables. | Public analytical tables contain required pseudonymous keys and no Riot identifier or credential. |
| FR22 | Offline validation shall verify raw checksums, schema, nested structures, match invariants, duplicate keys, rank joins, privacy, and feature leakage without contacting Riot or modifying raw data. | The command produces a validation report and exits non-zero when a critical issue exists. |
| FR23 | The sanitized validation layer shall be emitted only for `PASS` or `PASS_WITH_WARNINGS`. | A critical failure produces no validated dataset; a valid run contains only sanitized records named by its validated manifest. |

### 6.4 Splitting, feature, and modelling requirements

| ID | Requirement | Acceptance condition |
| --- | --- | --- |
| FR24 | The system shall consume the frozen chronological match-group split containing 3,500 training, 750 validation, and 750 test matches. | No match is duplicated, omitted, added, or divided across partitions, and the canonical assignment hash matches the contract. |
| FR25 | Grouped cross-validation shall keep every participant row from a match in one fold and shall report participant overlap separately. | Fold audits show zero match overlap; player-connected-component analysis may be used as a sensitivity check. |
| FR26 | Feature construction shall implement the cumulative F1 through F5 families defined in the analysis contract. | F1 contains basic board state; F2 adds units; F3 adds traits; F4 adds items; and F5 adds structural summaries. |
| FR27 | Feature vocabularies and frequency filtering shall be fitted on training data only, using a minimum category frequency of five. | Validation or test categories cannot create new columns, and the same fitted vocabulary yields a stable schema. |
| FR28 | The model matrix shall exclude identifiers, targets, placement-derived outcomes, rank snapshots, provenance, regions, timestamps, and prohibited progression fields. | A prohibited field in source predictors or encoded features aborts analysis. |
| FR29 | The top-four task shall train the registered dummy, logistic-regression, random-forest, XGBoost, LightGBM, and CatBoost candidates with fixed parameters and seed 4275. | The registry and emitted parameter records match the frozen analysis contract. |
| FR30 | The top-four model shall be selected by the highest validation F1 with registry order resolving an exact tie, then refitted on training plus validation data. | A locked selection manifest identifies one winner and contains no test-derived selection value. |
| FR31 | The exact-placement task shall compare only the registered dummy and random-forest candidates on validation MAE, minimising MAE and applying the declared tie-break. | A separate integrity-protected placement selection manifest identifies one winner. |
| FR32 | The test partition shall remain inaccessible until both task-specific selection manifests have been written and verified. | Early, missing, tampered, mismatched, or repeated access is rejected. |
| FR33 | Only the selected model for each task shall receive the final-test payload, and each task shall be evaluated at most once per production run. | The run manifest records one pre-payload access event per task and no unselected model has test metrics. |
| FR34 | The final evaluation shall report the metrics bound to each task and match-cluster bootstrap intervals for approved headline measures. | Resampling uses complete matches and the artifact set contains the required point estimates and interval bounds. |
| FR35 | Production analysis shall write the required manifests, split copy, feature vocabulary, validation and final-test tables, selected model files, and artifact checksums without overwriting an existing run. | The production artifact manifest is complete, every recorded hash verifies, and an existing destination causes failure. |

### 6.5 Reporting and dashboard requirements

| ID | Requirement | Acceptance condition |
| --- | --- | --- |
| FR36 | Post-model analyses shall read frozen production artifacts without changing them or initiating a new model fit or final-test decision. | Source hashes match before and after execution, and outputs are written to a separate versioned location. |
| FR37 | Exported tables and figures intended for reporting shall exclude prohibited identifiers and shall state whether evidence is descriptive, adjusted, predictive, or sensitivity-based. | Schema checks find no prohibited column and the artifact metadata identifies analytical scope. |
| FR38 | The dashboard shall load only allow-listed, aggregate artifacts and shall reproduce their stored values without recalculation. | Display loaders return values equal to frozen source files and reject an unexpected schema or path. |
| FR39 | The dashboard shall not import training or model libraries, load serialized models, write to frozen sources, or expose participant-level error tables. | Static and runtime tests confirm read-only, presentation-only behaviour. |
| FR40 | Missing optional dashboard artifacts shall disable only the affected content and display an unavailable message. | The remaining pages continue to load and no fallback modelling or synthetic result is generated. |
| FR41 | The dashboard shall distinguish validation, cross-validation, final-test, descriptive, and sensitivity results. | A metric cannot be loaded under a conflicting analytical scope. |

## 7 Non-functional requirements

| ID | Quality attribute | Requirement and measurable criterion |
| --- | --- | --- |
| NFR01 | Reproducibility | The same frozen inputs, configuration, dependency versions, and seed shall reproduce the same split, feature schema, selection decision, and deterministic outputs within the numerical behaviour of the pinned libraries. |
| NFR02 | Integrity | Raw files, manifests, configuration, split assignments, model selections, and published artifacts shall be protected by SHA-256 or equivalent stable hashes and verified before use. |
| NFR03 | Confidentiality | Credentials, PUUIDs, Riot IDs, raw responses, and retrievable private metadata shall remain outside logs, commits, dashboard sources, and publication artifacts. |
| NFR04 | Availability and recovery | Collection shall be resumable from checkpoints after transient failures, while completed raw responses remain immutable and terminal failed runs remain auditable. |
| NFR05 | Reliability | The system shall fail closed when a scientific contract, checksum, schema, cohort, split, feature, or test-access invariant cannot be proven. It shall not guess, repair, or substitute missing production evidence. |
| NFR06 | API stewardship | Requests shall respect configured throttling and server-provided limits, use bounded retries, and avoid uncontrolled retry loops. |
| NFR07 | Auditability | Every acceptance, rejection, recovery, model selection, and final-test access decision shall leave machine-readable evidence linked to a run identifier and timestamp. |
| NFR08 | Maintainability | Collection, validation, transformation, analysis, and presentation shall remain separate modules with version-controlled configuration and automated tests for critical invariants. |
| NFR09 | Portability | The supported environment shall be Python 3.12 on a system able to install the pinned dependencies; path handling shall be repository-relative rather than dependent on one user's directory. |
| NFR10 | Usability | Command-line failures shall identify the failed gate and return a non-zero status. The dashboard shall use clear study labels and shall expose limitations beside potentially misinterpreted results. |
| NFR11 | Performance | The collector shall process requests incrementally rather than retaining the full acquisition in memory, and the dashboard shall cache read-only artifact loads where this does not alter source data. |
| NFR12 | Scientific validity | Results shall be described as predictive associations from final-board observations. The software and presentation shall not represent them as causal effects or live-game utility. |

## 8 Data requirements

### 8.1 Data layers

The storage design shall maintain a clear boundary between private source data and publishable analytical evidence. Private storage contains raw Riot responses, identifier-bearing acquisition metadata, match identifiers, and rank-retrieval information. Interim storage contains pseudonymous relational tables. The validated layer contains sanitized match records and a manifest created only after the offline quality gate passes. Processed storage contains model-ready features, while model, report, output, and artifact directories contain fitted pipelines, evaluation records, and aggregate results.

Raw responses are append-only research records. Resuming a run may reuse a raw response only after checksum verification and validation under the current configuration. A prior acceptance decision must never be inherited merely because a raw file already exists.

### 8.2 Keys and cardinality

A match is represented by one pseudonymous `match_hash`. Each valid match has eight participant rows keyed by `match_hash` and `participant_hash`. Units, unit-held items, traits, and available augments use their parent keys plus deterministic within-record indices. The accepted main cohort has 5,000 match keys and 40,000 participant rows. The split table has one assignment per accepted match.

### 8.3 Targets and features

The binary target is derived as `placement <= 4`; exact placement retains the validated integer from one through eight. Targets must remain separate from predictors. Eligible primary features describe final-board level and size, unit presence and stars, trait presence and strength, item counts, and defined composition-structure summaries. Augments may be included only when present in the frozen schema; they must not be inferred or merged from an external source.

### 8.4 Retention and publication

Private raw data and metadata shall be access-controlled and retained according to the institutional data-management plan. Publication packages may contain source code, configuration templates, schemas, checksums, aggregate tables, figures, and sanitized fixtures. They shall not contain credentials, raw responses, PUUIDs, Riot IDs, or reversible lookup tables. Project-specific hashes remain pseudonyms and must not be described as anonymous data.

## 9 External interface requirements

### 9.1 Riot API interface

The client shall send authenticated HTTPS requests to the configured platform and regional routes. Endpoint-specific identifiers may be used in the request path, but logs shall record only endpoint templates, response status, timing, attempt count, and error class. The interface shall interpret rate-limit headers and `Retry-After` responses, distinguish retryable from terminal errors, and enforce timeouts.

### 9.2 Command-line interface

Collection, validation, normalisation, splitting, feature construction, training, preflight, and analysis shall be invocable as separate Python modules. Commands that operate on a collection shall require an explicit run identifier where ambiguity could affect provenance. The production analysis entry point shall require either an approved validation run or the explicitly non-analytical dry-run mode, but not both.

### 9.3 File interfaces

Structured outputs shall use documented JSON, CSV, YAML, Markdown, image, or serialized-model formats. Writes that establish research state shall be atomic. File paths recorded in manifests shall be relative to the repository or run root, validated against traversal, and accompanied by hashes where required by the contract.

### 9.4 Dashboard interface

The dashboard shall provide navigation across study design, feature families, model performance, interpretability, exact placement, reproducibility, composition evidence, robustness, and limitations. Filters may alter only the displayed subset of already generated aggregate data. The interface shall not provide training, threshold optimisation, live scoring, counterfactual recommendations, or raw-record exploration.

## 10 Verification and traceability

Requirements are verified through configuration validation, unit tests, integration tests, manifest reconciliation, checksum checks, static dashboard checks, and production preflight. Table 10.1 maps the principal requirement groups to their implementation and automated evidence.

**Table 10.1 Requirements traceability matrix**

| Requirement group | Primary implementation | Verification evidence |
| --- | --- | --- |
| FR01-FR05 | `src/config.py`, `src/provenance.py`, `analysis/src/contract.py`, `analysis/preflight.py` | `tests/test_config.py`, `tests/test_provenance.py`, `tests/analysis/test_analysis_contract.py` |
| FR06-FR09 | `src/collect_players.py`, `src/collect_match_ids.py`, `src/run_pipeline.py` | `tests/test_discovery_capacity.py`, `tests/test_multi_region.py` |
| FR10-FR12 | `src/api_client.py`, `src/collect_rank_snapshots.py` | `tests/test_api_client.py`, `tests/test_rank_snapshots.py` |
| FR13-FR15 | `src/collect_matches.py`, `src/utils.py`, `src/run_pipeline.py` | `tests/test_discovery_capacity.py`, `tests/test_provenance.py` |
| FR16-FR18 | `src/validate_matches.py`, `src/validation/validate_matches.py` | `tests/test_validate_matches.py`, `tests/validation/test_match_validation.py` |
| FR19-FR20 | `src/collect_rank_snapshots.py`, `src/recover_rank_snapshots.py` | `tests/test_rank_snapshots.py`, `tests/test_rank_recovery.py` |
| FR21-FR23 | `src/normalise_matches.py`, `src/validation/validate_collection.py` | `tests/test_offline_pipeline.py`, `tests/validation/test_collection_integration.py`, `tests/validation/test_rank_and_leakage.py` |
| FR24-FR28 | `src/create_splits.py`, `src/build_features.py`, `analysis/src/splits.py`, `analysis/src/features.py` | `tests/test_splits.py`, `tests/analysis/test_dataset_and_splits.py`, `tests/analysis/test_features_and_leakage.py` |
| FR29-FR35 | `analysis/src/models.py`, `analysis/run.py`, `analysis/src/gate.py` | `tests/analysis/test_models.py`, `tests/analysis/test_placement_selection.py`, `tests/analysis/test_analysis_contract.py` |
| FR36-FR37 | `analysis/post_model/`, `analysis/thesis_extension/`, `src/post_model/` | `tests/post_model/`, `tests/thesis_extension/`, `tests/composition_analysis/` |
| FR38-FR41 | `dashboard/app.py`, `dashboard/components/` | `tests/dashboard/test_dashboard.py` |

Verification of a production run also requires reconciliation of the configured counts and hashes with the run manifest, accepted cohort, validated manifest, split assignment, selection manifests, test-access record, and final artifact manifest. Passing isolated unit tests is necessary but does not replace the end-to-end production gates.

## 11 Acceptance criteria

The system is acceptable for thesis production only when all of the following conditions hold:

1. the collection and analysis release checks pass at their exact annotated tags;
2. the collection manifest identifies a completed 5,000-match cohort and its checksums reconcile;
3. offline validation reports `PASS` or `PASS_WITH_WARNINGS` with no critical finding;
4. the validated dataset contains 5,000 matches and 40,000 participant observations;
5. the authoritative 3,500/750/750 chronological match split verifies against its frozen hash;
6. feature construction produces the contract-defined schema without a prohibited predictor;
7. the primary and secondary selection manifests are present, independent, and intact before test access;
8. each selected model receives one recorded final-test evaluation and no unselected model receives the test payload;
9. every required production artifact exists and matches its manifest hash;
10. reporting outputs and dashboard sources contain no prohibited identifiers; and
11. the complete automated test suite passes in the pinned environment.

A failure of any condition prevents the affected run from being represented as thesis-valid evidence. Pilot runs, dry runs, partial collections, capacity failures, and superseded analysis contracts remain engineering or audit artifacts only.

## 12 Chapter summary

The requirements define a fail-closed research system whose central quality is traceable control over the path from live source responses to published findings. The system must acquire a precisely bounded cohort, preserve immutable evidence, remove direct identifiers, enforce structural and scientific validity, prevent feature and test leakage, and retain a complete record of analytical decisions. Its dashboard is intentionally limited to presenting frozen aggregate artifacts. These requirements align the software implementation with the study's observational scope and provide concrete tests for determining whether a collection or analysis run is suitable for inclusion in the thesis.

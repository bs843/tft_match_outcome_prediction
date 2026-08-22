# Data Dictionary

## Private acquisition tables

`seed_players.csv`, `match_id_sources.csv`, `unique_match_ids.csv`, `collection_status.csv`, `rank_snapshots.csv`, accepted/rejected manifests and raw JSON are private. They may contain private sampling metadata, PUUIDs or retrievable match identifiers and are not dissertation supplements.

`rank_snapshots.csv` contains exactly one row per unique accepted participant and run. It
stores a project-specific participant pseudonym, the attempted `RANKED_TFT` queue,
retrieval time, optional tier/division/league-point/win/loss values, and one status:

- `found`: a matching ranked entry was returned;
- `unranked`: the request succeeded but no matching ranked entry was returned;
- `not_found`: Riot returned HTTP 404 for that participant lookup;
- `request_failed`: the request still failed after the client's bounded retries.

`request_failed` rows are retained on normal reruns. Retrying them requires the explicit
`collect_rank_snapshots --retry-failed` option before the dataset is frozen.

## Model-facing relational tables

| Table | Key | Contents | Feature eligibility |
|---|---|---|---|
| `matches` | `match_hash` | Date, length, validated patch/set/queue and provenance | Patch is constant; timestamp, length and provenance excluded from primary model |
| `participants` | `match_hash, participant_hash` | Targets, final state, rank-snapshot audit fields and provenance | Only targets are used for labels; progression, rank snapshots and identifiers are prohibited |
| `units` | Participant key + `unit_index` | Character, rarity, stars and item count | Character and star information eligible |
| `unit_items` | Unit key + `item_index` | Item ID/name | Eligible |
| `traits` | Participant key + `trait_index` | Trait counts and activation | Active traits eligible |
| `augments` | Participant key + `augment_index` | Augment identifier | Eligible |

## Targets

- `placement`: validated integer 1–8; secondary ordinal target.
- `top_four`: boolean derived only from placement; primary target.
- `placement_group`: two-position grouping; secondary target.

## Prohibited model inputs

`placement`, `top_four`, `placement_group`, `last_round`, `players_eliminated`, `total_damage_to_players`, `time_eliminated`, all rank snapshot fields, match/participant hashes, retrieval times, run IDs, dataset versions and schema versions. Rank snapshots are available only for description, subgroup auditing or a separately declared sensitivity analysis.

## Missing values and types

Empty API fields are represented as null/empty values in relational outputs and become explicit zeros only where absence has a defined compositional meaning. CSV booleans are lowercase `true`/`false`. UTC timestamps use ISO 8601; source epoch times retain their declared millisecond unit.

The quality report records Riot's raw `info.endOfGameResult` value in `end_result`
and records `info.endOfGameResult` in `end_result_source`. Match acceptance requires
the observed value `GameComplete`; it is never filled with a constant.

## Post-collection validation artifacts

`reports/validation/<run-id>/validation_summary.json` is the authoritative run-level
gate result. Supporting CSV files record missingness, schema drift, invalid or
excluded records, end-result issues, duplicate keys and rank-snapshot join results.
Exception tables use keyed match/participant hashes only.

`data/validated/<run-id>/matches/*.json` contains privacy-sanitized copies created
only after a run passes the critical checks. Raw match IDs and PUUIDs are replaced by
project hashes, Riot display-name fields are removed, and `top_four_derived` is
reconstructed from validated placement. `validated_manifest.csv` records the source
checksum and validation provenance without exposing the source identifier.

## Collection-capacity reports

`player_sampling_report.csv` records aggregate ladder entries returned, unique players
available/selected, duplicate player entries removed and requested/populated/exhausted
pages for each run. `player_sampling_pages.csv` provides identifier-free tier/division/page
diagnostics, including entries returned and pagination exhaustion. `candidate_discovery_report.csv`
records raw candidate occurrences, unique seed-match associations, globally unique
candidate match IDs, duplicates removed/rate, the 25,000 threshold and gate result.
`reports/collection/collection_<run-id>.json`
combines those statistics with download counts, rejection counts by reason, acceptance
rate, rank-snapshot coverage and Git/configuration provenance. These aggregate reports
contain no Riot IDs, PUUIDs or match IDs.

`existing_verified` means an immutable raw response matched a previously recorded
SHA-256 digest and was parsed and revalidated for the current run. It never carries
forward prior acceptance or dataset membership.
## Multi-region provenance amendment

For configuration v5, `seed_players.csv`, `match_id_sources.csv`,
`unique_match_ids.csv`, `collection_status.csv`, `accepted_match_ids.csv`, and
`rank_snapshots.csv` carry explicit `platform_region` and `regional_route` fields.
Candidate source records also carry discovery rank/division and source type. The
candidate report contains one global row and one diagnostic row per platform.

Rank status semantics remain explicit: `unranked` is a successful lookup without a
ranked TFT entry; `not_found` means the player is absent from the expected platform;
`request_failed` records an API/routing failure and its error class.

# TFT Patch 17.8 MSc Research Pipeline

This repository collects and analyses **offline, completed** ranked Teamfight Tactics matches for the MSc study:

> Among ranked standard TFT Set 17 matches on patch 17.8, how accurately can final-board composition features classify a participant as finishing in the top four?

Collection configuration v7 broadens player discovery across ME1, EUW1, EUN1,
NA1 and KR while keeping Patch 17.8 and GOLD–MASTER fixed. Platform-to-cluster
routing and sampling limits are centralized under `platforms` in
`config/config.yaml`; see `docs/MULTI_REGION_AMENDMENT.md`.

It is not a live-game prediction system. Raw Riot responses are immutable and private; analytical tables contain keyed pseudonyms rather than Riot identifiers.

## Pipeline guarantees

- Public TFT patch 17.8 is kept distinct from its observed API client build
  `16.15`; exact inclusion validates `game_version` against client patch 16.15.
- Set 17, ranked-queue, completed-game, eight-participant validation.
- Timestamp-prefiltered and paginated match discovery.
- Conservative proactive throttling plus global `Retry-After` handling.
- Resumable atomic writes and SHA-256 raw-file verification.
- Request logs contain endpoint templates, not PUUIDs or credentials.
- All eight participants from a match remain in the same temporal split.
- Feature vocabularies are learned from the training split only.
- Outcome/progression fields are blocked from the model matrix.
- Final evaluation uses a frozen test split and match-cluster bootstrap intervals.

## Setup

Python 3.12 is required.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.lock
Copy-Item .env.example .env
```

Populate `.env` locally. Never paste credentials into configuration, code, logs, commits, or dissertation appendices.

```text
RIOT_API_KEY=RGAPI-...
PARTICIPANT_HASH_KEY=<at-least-32-random-bytes>
```

Generate the pseudonymisation key with:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

## Before collection

In `config/config.yaml`:

1. Verify every enabled `platform_region`/`regional_route` mapping with a minimal authenticated request.
2. Set `scope.start_time` and `scope.end_time` to verified UTC Unix-second boundaries for patch 17.8.
3. Confirm client patch `16.15`, queue ID `1100`, and `tft_game_type` from pilot responses.
4. Do not weaken the patch, set, or queue rejection gates.

Development keys expire regularly and personal keys have restrictive rate limits. The collector is intentionally conservative and may take hours for the main dataset.

If Riot returns HTTP 401/403, the collector pauses for up to 24 hours and checks
the ignored local `.env` every 30 seconds. Replace `RIOT_API_KEY` there with a new
key; the running process loads only a changed, non-empty value and retries the
interrupted request. Keys are never printed or written to request logs. Pressing
`Ctrl+C` during collection records the run as `partial` audit evidence.

## Run the pilot

For ladder sampling:

```powershell
python -m src.run_pipeline --target pilot --player-mode ladder
```

For manually supplied Riot IDs or PUUIDs, populate the private `data/input_seed_players.csv` and run:

```powershell
python -m src.run_pipeline --target pilot --player-mode manual
```

Add `--train` only after reviewing the pilot quality and split reports:

```powershell
python -m src.run_pipeline --target pilot --player-mode ladder --train
```

The command prints a run ID. Every private collection table, manifest, and experiment is linked to this ID.

## Run the frozen final collection

The main collection is bound to the annotated tag
`tft-patch17.8-final-collection-v8`. Before any API request, the pipeline verifies
that `HEAD` resolves to the tag's exact commit and that no tracked file is modified.
Untracked runtime files correctly excluded by `.gitignore` do not invalidate the run.

Verify the frozen state and start a new 5,000-match run with:

```powershell
git status
git rev-parse HEAD
git rev-parse "tft-patch17.8-final-collection-v8^{commit}"
python -m src.run_pipeline --target main --player-mode ladder
```

Do not reuse the interrupted runs `20260811T174646Z-34d8ca25` or
`20260820T152821Z-9df0ecbb`, `20260820T212450Z-31431b86`, or
`20260822T192049Z-84373556`, or `20260823T104747Z-7fee1987`. A main run rejects an explicit run ID and aborts
before collection if the tag is missing, points elsewhere, or tracked files are dirty.
Any behaviour-changing amendment requires a reviewed commit and a new version/tag.

### Authorized rank-snapshot recovery

Run `20260823T124729Z-673b6e6c` completed and froze its 5,000-match acquisition
but failed during rank routing because one transferred participant appeared first
on ME1 and later on EUW1. Configuration v8 authorizes only a downstream recovery
bound to the frozen cohort hashes. Review the routing-only gate first:

```powershell
python -m src.recover_rank_snapshots --run-id 20260823T124729Z-673b6e6c --dry-run
```

After review, the same command without `--dry-run` collects/reuses rank snapshots,
materializes downstream tables/splits/features and performs final validation. Its
module allowlist contains no player, candidate or match collection stage.

The v4 amendment preserves the GOLD, PLATINUM, EMERALD, DIAMOND and MASTER strata
and all eligibility rules. It reads three ladder pages per non-apex division, selects
up to 160 unique players per populated page, and retains three match-history pages.
The 25,000-unique-candidate requirement is a pre-download capacity gate, never a
match-validity threshold. A shortfall terminates the run as `capacity_failed` before
raw match download, acceptance, rank enrichment or analytical output. Discovery reports
record returned ladder entries, selected unique players, candidate occurrences,
unique match IDs and duplicates removed. Candidate IDs are deduplicated before any
download or acceptance accounting.

An `existing_verified` raw response is reused only after its SHA-256 digest agrees
with recorded provenance and the JSON is revalidated under the current configuration.
No prior run's acceptance decision, manifest membership or analytical status is inherited.

## Run stages independently

```powershell
python -m src.collect_players --run-id <run-id> --mode ladder
python -m src.collect_match_ids --run-id <run-id>
python -m src.collect_matches --run-id <run-id> --accepted-target 100
python -m src.validate_matches --run-id <run-id> --accepted-target 100
python -m src.collect_rank_snapshots --run-id <run-id>
python -m src.normalise_matches --run-id <run-id>
python -m src.create_splits --run-id <run-id>
python -m src.build_features --run-id <run-id>
python -m src.train_model --run-id <run-id> --experiment-id <experiment-id>
```

This separation allows transformations and modelling to be rerun without contacting Riot once the raw snapshot exists.

## Read-only post-collection validation

After collection has finished, run the offline validation gate before normalisation,
splitting, feature construction or modelling:

```powershell
python -m src.validation.validate_collection
```

With no argument, the command selects the latest manifest whose status is `complete`.
To audit a specific run, including an interrupted run, supply its literal run ID:

```powershell
python -m src.validation.validate_collection --run-id 20260810T113946Z-7065d209
```

The validator never calls Riot and never edits raw JSON or collection metadata. It
checks recorded SHA-256 checksums, schema and nested structures, Patch 17.8 scope,
match completion, participant/placement consistency, rank-snapshot joins, duplicate
keys, privacy and prohibited-feature leakage. Reports are written to
`reports/validation/<run-id>/`. A sanitized dataset is written to
`data/validated/<run-id>/` only when no critical failure exists. `FAIL` exits with
status code 1; interrupted runs are audit-only and cannot produce a validated dataset.

## Data layers

- `data/private/raw/patch_17_8/`: immutable API JSON; never publish.
- `data/private/metadata/`: PUUID-bearing collection state; never publish.
- `data/manifests/`: configuration hashes and collection provenance.
- `data/interim/<run-id>/`: pseudonymous relational tables.
- `data/splits/<run-id>/match_splits_<dataset-version>.csv`: versioned, frozen chronological match assignments. Older dataset-version manifests are retained unchanged.
- `data/processed/<run-id>/features.csv`: leakage-controlled model matrix.
- `data/validated/<run-id>/`: privacy-sanitized, validation-gated JSON and manifest;
  created only for a run with no critical validation failures.
- `data/private/metadata/rank_snapshots.csv`: one timestamped ranked-status row per
  unique accepted participant, including unsuccessful and unranked outcomes.
- `models/<experiment-id>/`: fitted pipeline, metrics and test predictions.
- `reports/validation/<run-id>/`: machine-readable validation summary, issue and
  exclusion tables, schema report, and human-readable report.
- `reports/`: other quality and split-audit reports.

Reproducibility means rebuilding results from the frozen raw snapshot and manifests. Re-querying a live API later is not expected to reconstruct the same sample.

For a completed main run, `collection_run_<run-id>.json` records the full Git SHA,
immutable tag, dataset/schema versions, Patch 17.8, requested target, actual start and
completion status/timestamps, counts, and the checksum of that run's accepted-match
partition. The post-collection validation report repeats this provenance chain.

## Tests

Tests never call the live Riot API:

```powershell
pytest
```

The main collection is permitted only after all tests pass and a 100-accepted-match pilot confirms routing, queue semantics, missingness, request budget, and zero identifier leakage.

## Research documentation

- `docs/METHODOLOGY.md`: sampling, leakage controls and statistical analysis.
- `docs/DATA_DICTIONARY.md`: table contracts and feature eligibility.
- `docs/DATA_MANAGEMENT.md`: privacy, retention and publication rules.
- `docs/REPRODUCIBILITY.md`: exact artifact and experiment requirements.

Riot's current API and TFT policies remain authoritative and must be rechecked immediately before collection.

## Post-collection thesis analysis

The offline, fail-closed downstream workflow is documented in
[`analysis/README.md`](analysis/README.md) and
[`docs/ANALYSIS_PIPELINE.md`](docs/ANALYSIS_PIPELINE.md). Once a 5,000-match run
has completed and passed post-collection validation, start it with:

```powershell
python -m analysis.run --validation-run ACTUAL_RUN_ID
```

`python -m analysis.run --dry-run` uses synthetic engineering records only and
cannot produce thesis findings.

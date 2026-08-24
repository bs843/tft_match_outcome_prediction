# Reproducibility Checklist

For every reported result retain:

- Frozen study configuration and SHA-256-equivalent configuration hash.
- Git commit or a checksum of the submitted source archive.
- Python 3.12 version and exact dependency lock.
- Collection run ID, start/end time and route assumptions.
- Full 40-character Git commit SHA and immutable annotated collection tag.
- Raw-response checksum manifest.
- Post-collection `validation_summary.json`, `schema_report.json`, issue/exclusion
  tables and human-readable validation report for the same run ID.
- Validation code version and validation timestamp. A `FAIL` report must be retained
  as an audit artifact but must not be used as an analytical dataset.
- Run-partition identifier and SHA-256 checksum for the frozen accepted-match manifest.
- Accepted and rejected match manifests.
- Rank-snapshot partition checksum, retrieval-status counts and retrieval timestamps.
- Dataset and schema versions.
- Frozen match split manifest and overlap audit.
- Dataset-versioned split filenames so protocol amendments never overwrite an earlier frozen split.
- Training-only feature vocabulary.
- Random seed and model hyperparameters.
- Cross-validation, validation and one-time test metrics.
- Match-cluster bootstrap intervals.
- Fitted preprocessing/model pipeline and pseudonymous test predictions.

Continuous integration uses sanitized fixtures and mocked HTTP responses. It must never require a Riot credential or contact the live API.

Fresh API collection is not an exact reproduction mechanism because the source service is dynamic. Exact reproduction begins with the archived, checksum-verified raw snapshot.

The post-collection gate is reproducible offline with:

```powershell
python -m src.validation.validate_collection --run-id <literal-run-id>
```

The raw checksum result in the report establishes that validation did not silently
change or substitute source responses. Only `PASS` or `PASS_WITH_WARNINGS` creates
`data/validated/<run-id>/`; critical failures return a non-zero process status.

The rank-recovery amendment tag is `tft-patch17.8-final-collection-v9`. At main-run startup,
the pipeline requires `HEAD` to equal that tag's peeled commit and rejects all
uncommitted changes to tracked files. The tag must never be moved or recreated after
collection begins. A later pipeline amendment requires a new commit and tag.
The earlier `tft-patch17.8-final-collection-v2` and
`tft-patch17.8-final-collection-v3` tags remain immutable audit evidence.
Run `20260820T152821Z-9df0ecbb` is a failed, exhausted-pool run and cannot be resumed,
merged, or used to create analytical output.
Run `20260820T212450Z-31431b86` is likewise permanently failed/audit-only. V4 main
runs reject supplied run IDs and enforce the 25,000-unique-candidate gate before raw
match download. `capacity_failed` is a terminal manifest status with no frozen accepted
partition or downstream analytical output.
## Multi-region provenance

Configuration v7 manifests record all enabled platform/route mappings. Seed,
candidate, status, accepted-manifest and rank-snapshot partitions retain structured
region provenance. The accepted-manifest checksum covers the enriched provenance
for new runs while retaining legacy checksum compatibility for earlier audit runs.

Candidate platform and download route are derived from the Riot match-ID prefix,
not assumed from the seed platform used for cluster-wide history discovery. Source
platform fields continue to record how a candidate was found. Candidate prefixes
outside the frozen enabled-platform list are counted and excluded before the global
capacity gate.

## Credential expiry during collection

On HTTP 401, or a 403 confirmed by a failed credential-health probe, the current
request pauses for at most 24 hours. The process checks the ignored `.env` every 30
seconds and resumes only after `RIOT_API_KEY` changes to a non-empty value. A 403
limited to `/tft/league/v1/by-puuid/{puuid}` is tested against the non-private GOLD I
ladder endpoint on the same platform. A successful probe proves the credential is
healthy; the participant row is checkpointed as `request_failed` with
`RiotForbiddenError`, allowing recovery to continue. Neither credentials nor PUUIDs
are logged. If credential health cannot be established, authentication fails closed.
`Ctrl+C` remains available and changes the manifest to terminal status `partial`;
partial and failed main runs are audit-only and are not resumed or merged.
Run `20260823T104747Z-7fee1987` was deliberately interrupted during discovery and
is preserved with terminal status `partial`.

## Downstream-only rank recovery

The `rank_snapshot_recovery_v3` amendment is narrowly authorized for run
`20260823T124729Z-673b6e6c`. It verifies the frozen 5,000-row accepted partition,
match-ID set, accepted CSV, and every accepted raw checksum before and after work.
It cannot invoke seed, candidate, or match collection. The original failure remains
in manifest history and a linked recovery manifest records the transition.

Participants may legitimately appear under more than one Riot platform within the
frozen observation window because of account/platform transfers. Rank snapshots are
participant-level and are routed through the platform associated with the participant's
most recent accepted match. Earlier platform observations are retained as provenance
and do not generate additional rank snapshot rows. Equal match timestamps use the
lexicographically greatest immutable match ID as the deterministic tie-break.
Routing remains `rank_snapshot_routing_v2`; recovery v3 changes only ambiguous-403
handling and uses a new manifest name so the interrupted v2 recovery record remains
immutable. Existing successful snapshot rows are reused, and only rows already
marked `request_failed` are retried.

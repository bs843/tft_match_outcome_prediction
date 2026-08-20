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

The amended final collection tag is `tft-patch17.8-final-collection-v3`. At main-run startup,
the pipeline requires `HEAD` to equal that tag's peeled commit and rejects all
uncommitted changes to tracked files. The tag must never be moved or recreated after
collection begins. A later pipeline amendment requires a new commit and tag.
The earlier `tft-patch17.8-final-collection-v2` tag remains immutable audit evidence.
Run `20260820T152821Z-9df0ecbb` is a failed, exhausted-pool run and cannot be resumed,
merged, or used to create analytical output.

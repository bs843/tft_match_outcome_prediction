# Read-only post-model analysis

## Master's-level evidence synthesis

The versioned evidence-synthesis layer integrates descriptive association,
statistical inference, training-only adjusted association, incremental validation
performance, frozen-model TreeSHAP interpretation, and predeclared sensitivity
checks. It does not alter the primary experiment, retrain a model, change model
selection, or use final-test outcomes for new feature discovery.

Authoritative outputs are written to
`artifacts/post_model/evidence_synthesis/`. The synthesis consumes the existing
identifier-free `composition_analysis_v1` tables and verifies that their hashes
remain unchanged. A train-versus-validation descriptive stability analysis may
read the frozen private cohort in memory, but exports only aggregates and never
uses test outcomes.

Run from the repository root with the full analysis environment:

```powershell
python -m src.post_model.evidence_synthesis
```

The command fails rather than overwriting an existing synthesis directory.


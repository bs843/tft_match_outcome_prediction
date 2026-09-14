# TFT MSc thesis dashboard

This directory contains a read-only Streamlit companion to the frozen Teamfight
Tactics thesis experiment. It presents the study design, cumulative feature-family
results, locked XGBoost final-test performance, approved model interpretation,
secondary exact-placement results, and reproducibility evidence.

The dashboard is a presentation layer only. It does not train, fit, refit, tune,
cross-validate, select, score, or load a model. It does not regenerate predictions or
confidence intervals. The final test set is not used for tuning or any new decision.

## Launch

From the repository root, install the dashboard dependency and run:

```bash
python -m pip install -r requirements.txt
streamlit run dashboard/app.py
```

The application uses programmatic Streamlit navigation and a wide academic layout.

## Pages

1. **Thesis overview** — frozen 5,000-match / 40,000-participant design, chronological
   3,500 / 750 / 750 split, outcome balance, placement balance, and approved aggregate
   top-four/bottom-four descriptions.
2. **Feature families** — cumulative F1–F5 validation performance and the frozen
   development candidate-model comparison.
3. **Model performance** — locked XGBoost final-test metrics, match-cluster bootstrap
   intervals, confusion matrix, ROC and precision–recall curves.
4. **Interpretability** — frozen TreeSHAP and grouped permutation-importance outputs,
   with explicit non-causal interpretation warnings.
5. **Exact placement** — secondary locked Random Forest results for placements 1–8.
6. **Reproducibility** — contract, production, split, hash, safety, and optional
   rank-stratified descriptive evidence.
7. **Top-four composition analysis** — support-filtered descriptive end-state
   associations, exploratory interactions, rank context, and locked TreeSHAP.
8. **Composition evidence** — integrated raw, adjusted, and predictive evidence
   with presentation-only filters and explicit evidence classifications.
9. **Robustness and limitations** — support-threshold stability, BH multiplicity
   summaries, train/validation directional stability, and the structured bias matrix.

## Frozen sources

The complete page-to-source provenance mapping is in
[`ARTIFACT_MAP.md`](ARTIFACT_MAP.md). Central path resolution and loading live in
`components/loaders.py`; paths are derived from the repository root with `pathlib`.
CSV, JSON, YAML, Markdown, and images are loaded with read-only APIs and immutable
tables are cached with `st.cache_data`.

The dashboard uses:

- `config/analysis_contract.yaml` for design and governance;
- `artifacts/post_model/` for frozen descriptive, performance, diagnostic, and
  verification outputs;
- `artifacts/thesis_extension_v1/` for frozen F1–F5, SHAP, uncertainty, and exact-
  placement outputs;
- `artifacts/post_model/evidence_synthesis/` for versioned evidence integration and
  sensitivity tables generated without retraining or final-test discovery;
- small production run and selection manifests under `outputs/` for authoritative
  model-lock and test-access metadata.

It intentionally does not load raw participant data, match-level assignments,
serialized models, or identifier-bearing error tables.

## Primary and secondary analyses

Top-four versus bottom-four classification is the primary task. XGBoost with the
frozen F5 feature family is the locked production model, and F1 is the primary
selection metric. Exact placement is labelled throughout as a secondary analysis.
Rank-stratified metrics are contextual descriptions only and rank snapshots remain
excluded from the primary predictive matrix.

Augments were planned "where available" but are absent from the frozen participant
schema. The dashboard does not infer, scrape, or externally merge augment values.

## Fail-safe behaviour

Missing artifacts produce a visible warning and disable only the affected section.
There is no modelling fallback. Display frames are checked for prohibited identifier
columns, and metric loaders enforce validation/final-test scope separation.

## Tests

Dashboard non-UI tests cover artifact resolution, schema and metric scope, prohibited
columns, read-only loading, static training-import guards, study totals, exact source
agreement, integrity verification, and graceful optional-artifact failure.

```bash
pytest tests/dashboard
```

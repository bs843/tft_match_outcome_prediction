"""Research integrity, provenance, artifact verification, and rank context."""

from __future__ import annotations

import streamlit as st

from dashboard.components.charts import format_metric_table
from dashboard.components.formatting import artifact_warning, method_note, section_kicker
from dashboard.components.loaders import (
    load_analysis_contract,
    load_dashboard_manifest,
    load_dataset_summary,
    load_production_manifest,
    load_rank_stratified_metrics,
    load_selection_manifest,
    load_verification_report,
    verify_post_model_bundle,
)
from dashboard.components.validation import DashboardDataError


def short_hash(value: object) -> str:
    text = str(value)
    return f"{text[:12]}…" if len(text) > 12 else text


section_kicker("Research integrity · provenance and validation")
st.title("Reproducibility and frozen-artifact status")
method_note(
    "The dashboard reads provenance and verification metadata. It does not repair, recreate, "
    "or replace any missing or changed production artifact."
)

try:
    contract = load_analysis_contract()
    dataset = dict(
        zip(
            load_dataset_summary()["measure"],
            load_dataset_summary()["value"],
            strict=False,
        )
    )
    production = load_production_manifest()
    selection = load_selection_manifest()
    verification = load_verification_report()
    bundle = verify_post_model_bundle()
    dashboard_manifest = load_dashboard_manifest()

    columns = st.columns(4)
    columns[0].metric("Production status", str(production["status"]).replace("_", " ").title())
    columns[1].metric("Recorded production verification", verification["status"])
    columns[2].metric("Live post-model bundle integrity", bundle["status"])
    test_value = dashboard_manifest.get("tests", {}).get("passed")
    columns[3].metric("Tests passed", str(test_value) if test_value is not None else "Not recorded")

    integrity_rows = [
        ("Dataset version", verification["dataset_version"]),
        ("Patch", contract["dataset"]["patch"]),
        ("Analysis-contract version", verification["analysis_contract"]),
        ("Analysis tag", verification["analysis_tag"]),
        ("Production run", verification["production_run_id"]),
        ("Locked primary model", str(selection["selected_model"]).upper()),
        ("Frozen matches", int(float(dataset["accepted_matches"]))),
        ("Participant rows", int(float(dataset["participant_rows"]))),
        ("Train / validation / test", "3,500 / 750 / 750"),
        ("Production hashes verified", verification["artifact_hashes_verified"]),
        ("Post-model files verified now", f"{bundle['verified']} / {bundle['expected']}"),
        ("Retraining during post-model analysis", "NO"),
        ("Final-test tuning", "NO"),
    ]
    st.dataframe(
        {
            "Integrity item": [row[0] for row in integrity_rows],
            "Recorded value": [str(row[1]) for row in integrity_rows],
        },
        hide_index=True,
        width="stretch",
    )

    if bundle["status"] != "PASS":
        st.error("Frozen post-model bundle verification failed. Dashboard trust indicators are disabled.")
        for failure in bundle["failures"]:
            st.write(f"- {failure}")

    st.subheader("Safe provenance hashes")
    hashes = {
        "Frozen split assignment": short_hash(production["split_hash"]),
        "Analysis contract": short_hash(production["configuration_hashes"]["analysis_contract"]),
        "Feature schema": short_hash(verification["feature_manifest_hash"]),
        "Encoded F5 identity": short_hash(verification["encoded_feature_identity_sha256"]),
        "Locked selection manifest": short_hash(selection["manifest_sha256"]),
    }
    st.dataframe(
        {"Artifact / configuration": list(hashes), "SHA-256 (truncated)": list(hashes.values())},
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Only cryptographic provenance hashes for files, configurations, splits, and models are "
        "shown. Participant and match identifiers are never loaded into the dashboard."
    )
except (DashboardDataError, KeyError, TypeError, ValueError) as error:
    artifact_warning(error, "Reproducibility summary")

st.subheader("Governance checks")
checks = {
    "Training invoked by dashboard": "NO",
    "Models refit": "NO",
    "Model selection changed": "NO",
    "Hyperparameters changed": "NO",
    "Split assignments changed": "NO",
    "Final-test tuning introduced": "NO",
    "Rank variables in primary predictive matrix": "NO",
    "Participant or match identifiers displayed": "NO",
}
st.dataframe(
    {"Check": list(checks), "Status": list(checks.values())},
    hide_index=True,
    width="stretch",
)

st.divider()
st.subheader("Rank-stratified descriptive analysis")
method_note(
    "Rank data is retained for contextual analysis but excluded from the primary predictive "
    "matrix in accordance with the frozen analysis contract. Stratum metrics are descriptive "
    "final-test summaries, not a basis for model selection."
)
try:
    rank = load_rank_stratified_metrics()
    if rank is None:
        st.warning(
            "Approved frozen rank-stratified output was not found, so this optional section "
            "has been disabled."
        )
    else:
        display = format_metric_table(
            rank,
            ["rank_stratum", "n", "matches", "prevalence", "accuracy", "f1", "roc_auc", "pr_auc"],
        ).rename(
            columns={
                "rank_stratum": "Rank stratum",
                "n": "Participant observations",
                "matches": "Matches represented",
                "prevalence": "Top-four rate",
                "accuracy": "Accuracy",
                "f1": "F1",
                "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC",
            }
        )
        st.dataframe(display, hide_index=True, width="stretch")
        rank_chart = rank.set_index("rank_stratum")[["f1"]].rename(columns={"f1": "Descriptive F1"})
        st.bar_chart(rank_chart, color="#6f8798")
        st.caption(
            "Small strata can yield unstable metrics; sample sizes should be considered before "
            "interpreting between-rank differences."
        )
except DashboardDataError as error:
    artifact_warning(error, "Frozen rank-stratified analysis")

"""Study design, outcome balance, and approved aggregate exploration."""

from __future__ import annotations

import streamlit as st

from dashboard.components.charts import FEATURE_LABELS, outcome_comparison
from dashboard.components.formatting import artifact_warning, method_note, section_kicker
from dashboard.components.loaders import (
    load_analysis_contract,
    load_class_distribution,
    load_dataset_summary,
    load_outcome_summary,
    load_placement_distribution,
    load_placement_selection_manifest,
    load_selection_manifest,
)
from dashboard.components.validation import DashboardDataError


section_kicker("MSc thesis analytical prototype")
st.title("Predicting Teamfight Tactics Match Outcomes Using End-State Composition Features")
st.write(
    "A read-only presentation of the frozen Patch 17.8 experiment: study design, "
    "composition–performance patterns, locked model results, and post-model interpretation."
)
method_note(
    "This dashboard consumes frozen outputs only. It does not retrain, refit, tune, "
    "reselect, or regenerate model predictions."
)

try:
    contract = load_analysis_contract()
    summary = load_dataset_summary()
    primary_selection = load_selection_manifest()
    placement_selection = load_placement_selection_manifest()
    measures = dict(zip(summary["measure"], summary["value"], strict=False))

    cols = st.columns(4)
    cols[0].metric("Completed matches", f"{int(float(measures['accepted_matches'])):,}")
    cols[1].metric("Participant observations", f"{int(float(measures['participant_rows'])):,}")
    cols[2].metric("Primary task", "Top-4 / bottom-4")
    cols[3].metric("Primary metric", str(contract["targets"]["top4"]["primary_metric"]).upper())

    cols = st.columns(4)
    cols[0].metric("Training matches", f"{int(contract['split']['counts']['train']):,}")
    cols[1].metric("Validation matches", f"{int(contract['split']['counts']['validation']):,}")
    cols[2].metric("Final-test matches", f"{int(contract['split']['counts']['test']):,}")
    cols[3].metric("Locked primary model", str(primary_selection["selected_model"]).upper())

    st.caption(
        "Secondary task: exact placement (1st–8th), with locked model "
        f"{str(placement_selection['selected_model']).replace('_', ' ').title()}."
    )
except (DashboardDataError, KeyError, TypeError, ValueError) as error:
    artifact_warning(error, "Study-design summary")

st.subheader("Frozen analytical workflow")
patch = "17.8"
try:
    patch = str(contract["dataset"]["patch"])
except (NameError, KeyError, TypeError):
    pass
st.markdown(
    f"""
    <div class="workflow">
    Riot TFT API &nbsp;→&nbsp; Frozen Patch {patch} dataset &nbsp;→&nbsp;
    End-state feature engineering &nbsp;→&nbsp; Train / validation / final-test split<br>
    Model selection &nbsp;→&nbsp; Locked XGBoost model &nbsp;→&nbsp;
    Frozen final-test evaluation &nbsp;→&nbsp; Post-model interpretation
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")
method_note(
    "All eight participants from each match remained in the same chronological partition, "
    "preventing match-level leakage. The final test set remained frozen throughout feature "
    "and model selection."
)

st.subheader("Outcome balance")
try:
    balance = load_class_distribution()
    left, right = st.columns([1.4, 1])
    with left:
        chart = balance.set_index("outcome")[["participants"]].rename(
            index={"bottom_four": "Bottom four", "top_four": "Top four"},
            columns={"participants": "Participant observations"},
        )
        st.bar_chart(chart, color="#315b7d")
    with right:
        display = balance.copy()
        display["percentage"] = (display["proportion"] * 100).map(lambda x: f"{x:.1f}%")
        display["outcome"] = display["outcome"].str.replace("_", " ").str.title()
        st.dataframe(
            display[["outcome", "participants", "percentage"]],
            hide_index=True,
            width="stretch",
        )
    st.caption(
        "The target is balanced by construction: each valid TFT match contributes four "
        "top-four and four bottom-four observations. No resampling or SMOTE was applied."
    )
except DashboardDataError as error:
    artifact_warning(error, "Frozen class distribution")

st.subheader("Final placement distribution")
try:
    placements = load_placement_distribution()
    chart = placements.set_index("placement")[["participants"]].rename(
        columns={"participants": "Participant observations"}
    )
    st.bar_chart(chart, color="#6f8798")
    st.caption(
        "Actual stored counts are shown for placements 1–8; the balanced structure follows "
        "from one participant occupying each placement in every accepted match."
    )
except DashboardDataError as error:
    artifact_warning(error, "Frozen placement distribution")

st.subheader("Top-four vs bottom-four descriptive patterns")
st.caption(
    "Select a thesis-relevant aggregate end-state variable. These are descriptive "
    "associations, not independent interventions or causal effects."
)
try:
    outcome = load_outcome_summary()
    available = [name for name in FEATURE_LABELS if name in set(outcome["variable"])]
    selected = st.selectbox(
        "End-state feature",
        available,
        format_func=lambda value: FEATURE_LABELS[value],
    )
    comparison = outcome_comparison(outcome, selected)
    left, right = st.columns([1.6, 1])
    with left:
        st.bar_chart(comparison, color=["#315b7d", "#9aafbd"])
    with right:
        st.dataframe(comparison.round(3), width="stretch")
    st.caption(
        f"{FEATURE_LABELS[selected]} differed descriptively between the stored outcome "
        "groups. The dashboard does not infer that changing this feature would cause a "
        "different placement."
    )
except (DashboardDataError, ValueError) as error:
    artifact_warning(error, "Approved outcome summaries")

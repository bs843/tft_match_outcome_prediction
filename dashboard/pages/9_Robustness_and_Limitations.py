"""Read-only robustness and limitations views for the evidence synthesis."""

from __future__ import annotations

import streamlit as st

from dashboard.components.formatting import artifact_warning, method_note, section_kicker
from dashboard.components.loaders import (
    load_evidence_synthesis_table,
    load_limitations_matrix,
)
from dashboard.components.validation import DashboardDataError


section_kicker("Sensitivity and claim governance")
st.title("Robustness and limitations")
method_note(
    "Sensitivity analyses do not alter the primary findings, select a model, or use the final "
    "test set for feature discovery. Rank is retrieval-time context only."
)

try:
    sensitivity = load_evidence_synthesis_table("support_threshold_sensitivity.csv")
    stability = load_evidence_synthesis_table("threshold_rank_stability.csv")
    testing = load_evidence_synthesis_table("multiple_testing_summary.csv")
    development = load_evidence_synthesis_table("train_validation_descriptive_stability.csv")
    limitations = load_limitations_matrix()
except DashboardDataError as error:
    artifact_warning(error, "Robustness evidence")
    st.stop()

view = st.segmented_control(
    "View", ["Support thresholds", "Multiple testing", "Development stability", "Limitations"],
    default="Support thresholds", key="robustness_view",
)

if view == "Support thresholds":
    feature_type = st.selectbox("Feature type", sorted(sensitivity.feature_type.unique()),
                                key="robustness_feature_type")
    selected = sensitivity.loc[sensitivity.feature_type == feature_type]
    st.dataframe(selected.sort_values(["support_threshold", "absolute_uplift_rank"]),
                 hide_index=True, key="support_sensitivity_table")
    st.subheader("Rank stability")
    st.dataframe(stability.loc[stability.feature_type == feature_type], hide_index=True,
                 key="support_rank_stability_table")
    st.caption("The interaction source retained only n≥150 combinations, so n=100 cannot be reconstructed without rerunning the exploratory search.")
elif view == "Multiple testing":
    st.dataframe(testing, hide_index=True, key="multiple_testing_table")
    st.caption("Benjamini–Hochberg q-values are the sole multiplicity correction. Practical magnitude is considered separately.")
elif view == "Development stability":
    feature_type = st.selectbox("Feature type", sorted(development.feature_type.unique()),
                                key="development_stability_type")
    selected = development.loc[development.feature_type == feature_type]
    same = selected.direction_same.mean()
    with st.container(horizontal=True):
        st.metric("Supported comparisons", f"{len(selected):,}", border=True)
        st.metric("Same direction", f"{same:.1%}", border=True)
        st.metric("Median absolute difference",
                  f"{selected.absolute_difference_pp.median():.2f} pp", border=True)
    st.dataframe(selected.sort_values("absolute_difference_pp", ascending=False),
                 hide_index=True, key="development_stability_table")
    st.caption("Only frozen train and validation partitions are compared; final-test outcomes are excluded.")
else:
    st.markdown(limitations)
    st.warning(
        "The dashboard describes and predicts final outcomes; it does not estimate what would "
        "have happened if a player had chosen a different unit, item or trait during the match."
    )


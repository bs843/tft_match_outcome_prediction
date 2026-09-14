"""Frozen SHAP and grouped predictive importance outputs."""

from __future__ import annotations

import streamlit as st

from dashboard.components.charts import add_feature_labels, format_metric_table
from dashboard.components.formatting import artifact_warning, method_note, section_kicker
from dashboard.components.loaders import (
    frozen_figure,
    load_feature_group_importance,
    load_feature_importance,
)
from dashboard.components.validation import DashboardDataError


section_kicker("Post-model interpretation · frozen approved outputs")
st.title("Predictive importance of end-state features")
method_note(
    "Feature importance identifies predictive association, not causal effect. End-state "
    "features describe the final board after the match has unfolded and must not be "
    "interpreted as independent causal interventions."
)

st.subheader("Global feature importance")
try:
    importance = add_feature_labels(load_feature_importance())
    top_n = st.select_slider("Number of features", options=[10, 20, 30], value=20)
    top = importance.head(top_n).sort_values("mean_absolute_shap", ascending=True)
    st.bar_chart(
        top,
        x="mean_absolute_shap",
        y="display_feature",
        horizontal=True,
        color="#315b7d",
    )
    table = format_metric_table(
        importance.head(top_n),
        ["display_feature", "substantive_unit", "mean_absolute_shap", "mean_signed_shap"],
    ).rename(
        columns={
            "display_feature": "Feature",
            "substantive_unit": "Feature group",
            "mean_absolute_shap": "Mean |SHAP|",
            "mean_signed_shap": "Mean signed SHAP",
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "TreeSHAP values are frozen validation-set log-odds contributions produced by the "
        "approved post-model analysis. Changing the selector changes presentation only."
    )
except (DashboardDataError, KeyError) as error:
    artifact_warning(error, "Frozen SHAP importance")

figure = frozen_figure("artifacts/thesis_extension_v1/figures/shap_feature_importance.png")
if figure:
    with st.expander("View frozen SHAP figure"):
        st.image(str(figure), width="stretch")
else:
    st.warning("Frozen SHAP figure was not found; tabular interpretability remains available.")

st.subheader("Feature-family importance")
try:
    groups = load_feature_group_importance().sort_values(
        "absolute_mean_importance_sum", ascending=True
    )
    st.bar_chart(
        groups,
        x="absolute_mean_importance_sum",
        y="feature_group",
        horizontal=True,
        color="#6f8798",
    )
    display = format_metric_table(
        groups.sort_values("absolute_mean_importance_sum", ascending=False),
        ["feature_group", "feature_count", "absolute_mean_importance_sum", "mean_absolute_importance"],
    ).rename(
        columns={
            "feature_group": "Conceptual group",
            "feature_count": "Encoded columns",
            "absolute_mean_importance_sum": "Summed absolute importance",
            "mean_absolute_importance": "Mean absolute importance",
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption(
        "This frozen grouped view sums absolute permutation importance within deterministic "
        "feature-prefix groups: basic end state, units/champions, traits, items, and "
        "composition structure."
    )
except DashboardDataError as error:
    artifact_warning(error, "Frozen grouped importance")

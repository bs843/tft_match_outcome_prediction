"""Frozen cumulative feature-family and development model comparisons."""

from __future__ import annotations

import streamlit as st

from dashboard.components.charts import feature_family_interpretation, format_metric_table
from dashboard.components.formatting import artifact_warning, method_note, section_kicker
from dashboard.components.loaders import (
    load_analysis_contract,
    load_feature_family_definitions,
    load_feature_family_metrics,
    load_validation_metrics,
)
from dashboard.components.validation import DashboardDataError


section_kicker("Development evidence · validation partition")
st.title("Cumulative feature-family performance")
st.write(
    "The five families were prespecified as cumulative representations of final-board state. "
    "This page reports the frozen logistic feature-family comparison used to describe the "
    "incremental predictive information in F1 through F5."
)
method_note(
    "All results on this page are development/validation results. They are not final-test "
    "performance and were not recalculated by the dashboard."
)

try:
    definitions = load_feature_family_definitions()
    metrics = load_feature_family_metrics()
    merged = definitions.merge(metrics, on="feature_family", how="inner")

    st.subheader("Family definitions")
    st.dataframe(
        merged[["feature_family", "included_units", "variables_or_patterns"]].rename(
            columns={
                "feature_family": "Feature family",
                "included_units": "Cumulative content",
                "variables_or_patterns": "Frozen variables / patterns",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Validation performance")
    table = format_metric_table(
        merged.rename(columns={"feature_family": "Feature family"}),
        [
            "Feature family",
            "model",
            "partition",
            "f1",
            "accuracy",
            "roc_auc",
            "pr_auc",
            "change_in_f1_from_previous_family",
        ],
    ).rename(
        columns={
            "model": "Model",
            "partition": "Partition",
            "f1": "F1",
            "accuracy": "Accuracy",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "change_in_f1_from_previous_family": "ΔF1 from previous",
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("F1 progression")
    progression = metrics.set_index("feature_family")[["f1"]].rename(columns={"f1": "Validation F1"})
    st.line_chart(progression, color="#315b7d")
    st.caption(
        "F1 is the headline metric because it was the frozen primary model-selection metric."
    )

    st.subheader("Neutral interpretation")
    for message in feature_family_interpretation(metrics):
        st.markdown(f"- {message}")
    contract = load_analysis_contract()
    st.write(
        f"The frozen contract designates **{contract['features']['primary_family']}** as the "
        "primary cumulative family. Incremental gains indicate predictive association, not "
        "causal importance."
    )
except (DashboardDataError, KeyError, TypeError, ValueError) as error:
    artifact_warning(error, "Frozen feature-family analysis")

st.divider()
st.subheader("Candidate model comparison")
st.caption(
    "Candidate selection used permitted development data only. The final test partition did "
    "not contribute to model selection."
)
try:
    models = load_validation_metrics().sort_values("f1", ascending=False)
    model_table = format_metric_table(
        models,
        ["model", "partition", "f1", "accuracy", "roc_auc", "pr_auc", "selected"],
    ).rename(
        columns={
            "model": "Model",
            "partition": "Partition",
            "f1": "F1",
            "accuracy": "Accuracy",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "selected": "Locked selection",
        }
    )
    model_table["Model"] = model_table["Model"].str.replace("_", " ").str.title()
    st.dataframe(model_table, hide_index=True, width="stretch")
    chart = models.set_index("model")[["f1"]].rename(columns={"f1": "Validation F1"})
    st.bar_chart(chart, color="#315b7d")
    method_note(
        "XGBoost was locked by the prespecified validation-F1 rule. LightGBM was the runner-up; "
        "the final test set was accessed only after the selection manifest was written."
    )
except DashboardDataError as error:
    artifact_warning(error, "Frozen validation model comparison")

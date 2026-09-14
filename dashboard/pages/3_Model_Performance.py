"""Locked XGBoost final-test performance and uncertainty."""

from __future__ import annotations

import streamlit as st

from dashboard.components.charts import format_metric_table
from dashboard.components.formatting import artifact_warning, method_note, section_kicker
from dashboard.components.loaders import (
    frozen_figure,
    load_confusion_matrix,
    load_cross_validation_metrics,
    load_final_test_metrics,
    load_metric_confidence_intervals,
    load_pr_points,
    load_roc_points,
)
from dashboard.components.validation import DashboardDataError


section_kicker("Primary result · frozen final test")
st.title("Locked XGBoost performance")
method_note(
    "These values come from the single frozen final-test evaluation of the already locked "
    "XGBoost F5 model. No threshold, feature, hyperparameter, or model decision is made here."
)

try:
    final = load_final_test_metrics()
    row = final.loc[final["target"] == "top4"].iloc[0]
    ci = load_metric_confidence_intervals().set_index("metric")
    f1_ci = ci.loc["f1"]

    columns = st.columns(5)
    columns[0].metric(
        "F1",
        f"{row['f1']:.3f}",
        help="Harmonic mean of precision and recall for top-four classification.",
    )
    columns[1].metric("Accuracy", f"{row['accuracy']:.3f}")
    columns[2].metric("ROC-AUC", f"{row['roc_auc']:.3f}")
    columns[3].metric("PR-AUC", f"{row['pr_auc']:.3f}")
    columns[4].metric(
        "F1 95% CI",
        f"{f1_ci['lower_95']:.3f}–{f1_ci['upper_95']:.3f}",
        help="Match-cluster bootstrap interval from 2,000 frozen resamples.",
    )
    st.caption(
        f"Model: {str(row['model']).upper()} · Feature family: {row['feature_family']} · "
        f"Partition: frozen {row['partition']}"
    )
except (DashboardDataError, IndexError, KeyError, TypeError, ValueError) as error:
    artifact_warning(error, "Frozen final-test metrics")

st.subheader("Match-cluster bootstrap confidence intervals")
try:
    ci_display = format_metric_table(
        load_metric_confidence_intervals(),
        ["metric", "point_estimate", "lower_95", "upper_95", "bootstrap_samples", "resampling_unit"],
    ).rename(
        columns={
            "metric": "Metric",
            "point_estimate": "Point estimate",
            "lower_95": "95% lower",
            "upper_95": "95% upper",
            "bootstrap_samples": "Bootstrap samples",
            "resampling_unit": "Resampling unit",
        }
    )
    st.dataframe(ci_display, hide_index=True, width="stretch")
    st.caption(
        "Participant observations within a match are not independent, so bootstrap resampling "
        "was performed at match level."
    )
except DashboardDataError as error:
    artifact_warning(error, "Frozen confidence intervals")

st.subheader("Confusion matrix")
try:
    matrix = load_confusion_matrix()
    left, right = st.columns([1, 1.3])
    with left:
        st.dataframe(matrix, width="stretch")
        st.caption("Rows are true outcomes; columns are locked-model predictions.")
    with right:
        figure = frozen_figure("artifacts/post_model/figures/final_test_confusion_matrix.png")
        if figure:
            st.image(str(figure), caption="Frozen final-test confusion matrix", width="stretch")
        else:
            st.warning("Frozen confusion-matrix figure was not found; the stored counts remain available.")
except DashboardDataError as error:
    artifact_warning(error, "Frozen confusion matrix")

st.subheader("Discrimination curves")
roc_col, pr_col = st.columns(2)
with roc_col:
    st.markdown("#### ROC curve")
    try:
        roc = load_roc_points()[["false_positive_rate", "true_positive_rate"]].rename(
            columns={
                "false_positive_rate": "False-positive rate",
                "true_positive_rate": "True-positive rate",
            }
        )
        st.line_chart(roc, x="False-positive rate", y="True-positive rate", color="#315b7d")
        st.caption("Frozen final-test ROC points; no scores were regenerated.")
    except DashboardDataError as error:
        artifact_warning(error, "Frozen ROC curve")
with pr_col:
    st.markdown("#### Precision–recall curve")
    try:
        pr = load_pr_points()[["recall", "precision"]].rename(
            columns={"recall": "Recall", "precision": "Precision"}
        )
        st.line_chart(pr, x="Recall", y="Precision", color="#6f8798")
        st.caption("Frozen final-test precision–recall points; no scores were regenerated.")
    except DashboardDataError as error:
        artifact_warning(error, "Frozen precision–recall curve")

st.subheader("Development cross-validation context")
try:
    cv = load_cross_validation_metrics().sort_values("cv_f1_mean", ascending=False)
    st.dataframe(
        format_metric_table(cv, ["model", "cv_f1_mean", "cv_f1_std", "cv_folds"]).rename(
            columns={
                "model": "Model",
                "cv_f1_mean": "Mean CV F1",
                "cv_f1_std": "CV F1 SD",
                "cv_folds": "Folds",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Cross-validation is shown as development context only. Selection followed the frozen "
        "validation-F1 rule, under which XGBoost was locked."
    )
except DashboardDataError as error:
    artifact_warning(error, "Frozen cross-validation summary")

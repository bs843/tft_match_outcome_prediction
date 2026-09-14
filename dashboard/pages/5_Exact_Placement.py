"""Secondary frozen exact-placement analysis."""

from __future__ import annotations

import streamlit as st

from dashboard.components.charts import format_metric_table
from dashboard.components.formatting import artifact_warning, method_note, section_kicker
from dashboard.components.loaders import (
    frozen_figure,
    load_exact_placement_ci,
    load_exact_placement_confusion,
    load_exact_placement_metrics,
)
from dashboard.components.validation import DashboardDataError


section_kicker("Secondary analysis · frozen final test")
st.title("Exact placement (1st–8th)")
method_note(
    "Exact placement is a secondary analysis. The displayed Random Forest was locked using "
    "validation MAE before the single read-only final-test evaluation.",
    secondary=True,
)

try:
    metrics = load_exact_placement_metrics()
    locked = metrics.loc[metrics["model"] == "random_forest_locked"].iloc[0]
    columns = st.columns(5)
    columns[0].metric("MAE", f"{locked['mae']:.3f}", help="Mean absolute placement error.")
    columns[1].metric("Exact accuracy", f"{locked['accuracy']:.3f}")
    columns[2].metric("Macro F1", f"{locked['macro_f1']:.3f}")
    columns[3].metric("Within ±1", f"{locked['within_1_accuracy']:.3f}")
    columns[4].metric("Within ±2", f"{locked['within_2_accuracy']:.3f}")
    st.write(
        f"An MAE of **{locked['mae']:.3f}** means that the model's predicted final "
        "placement differed from the true placement by just over one position on average."
    )
except (DashboardDataError, IndexError, KeyError, TypeError, ValueError) as error:
    artifact_warning(error, "Frozen exact-placement metrics")

st.subheader("Match-cluster uncertainty")
try:
    ci = load_exact_placement_ci()
    st.dataframe(
        format_metric_table(
            ci,
            ["metric", "estimate", "ci95_low", "ci95_high", "bootstrap_samples", "resampling_unit"],
        ).rename(
            columns={
                "metric": "Metric",
                "estimate": "Estimate",
                "ci95_low": "95% lower",
                "ci95_high": "95% upper",
                "bootstrap_samples": "Bootstrap samples",
                "resampling_unit": "Resampling unit",
            }
        ),
        hide_index=True,
        width="stretch",
    )
except DashboardDataError as error:
    artifact_warning(error, "Frozen exact-placement confidence intervals")

st.subheader("Placement confusion matrix")
try:
    matrix = load_exact_placement_confusion()
    st.dataframe(matrix, width="stretch")
    figure = frozen_figure(
        "artifacts/thesis_extension_v1/figures/placement_final_test_confusion_matrix.png"
    )
    if figure:
        st.image(str(figure), caption="Frozen 1st–8th placement confusion matrix", width="stretch")
    st.caption(
        "Rows are true placements and columns are locked-model predicted placements. "
        "Near-diagonal mass corresponds to predictions within one or two positions."
    )
except DashboardDataError as error:
    artifact_warning(error, "Frozen placement confusion matrix")

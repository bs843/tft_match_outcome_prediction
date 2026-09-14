"""Read-only integration of descriptive, adjusted and predictive evidence."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.formatting import artifact_warning, method_note, section_kicker
from dashboard.components.loaders import (
    load_evidence_synthesis_manifest,
    load_evidence_synthesis_table,
)
from dashboard.components.validation import DashboardDataError


section_kicker("Evidence synthesis · Frozen Patch 17.8 cohort")
st.title("Composition evidence")
method_note(
    "Raw uplift, adjusted odds ratios and TreeSHAP answer different questions and are not "
    "interchangeable. All controls below filter precomputed aggregate tables only."
)

try:
    evidence = load_evidence_synthesis_table("feature_evidence_table.csv")
    interactions = load_evidence_synthesis_table("interaction_strength_comparison.csv")
    manifest = load_evidence_synthesis_manifest()
except (DashboardDataError, KeyError) as error:
    artifact_warning(error, "Master's-level evidence synthesis")
    st.stop()

with st.container(horizontal=True):
    st.metric("Interpretable rows", f"{len(evidence):,}", border=True)
    st.metric("Mapped adjusted estimates", f"{evidence.adjusted_odds_ratio.notna().sum():,}", border=True)
    st.metric("Mapped SHAP features", f"{evidence.mean_abs_shap.notna().sum():,}", border=True)
    st.metric("Synthesis status", manifest["status"], border=True)

tabs = st.tabs(["Units", "Items", "Traits", "Interactions", "Structure"])
types = ["unit", "item", "trait_signature", "interaction", "composition_structure"]
for tab, feature_type in zip(tabs, types, strict=False):
    with tab:
        if feature_type == "interaction":
            frame = interactions.copy()
            frame["feature"] = frame.feature_a + " + " + frame.feature_b
            frame["support_n"] = frame.appearances
            frame["uplift_pp"] = frame.uplift_vs_global_pp
            frame["q_value"] = frame.fdr_bh_q_value
            columns = ["feature", "interaction_type", "support_n", "top_four_rate",
                       "uplift_pp", "top_four_rate_ci95_low", "top_four_rate_ci95_high",
                       "q_value", "interaction_excess_over_best_component_pp", "interpretation"]
        else:
            frame = evidence.loc[evidence.feature_type == feature_type].copy()
            columns = ["feature", "support_n", "top4_rate", "uplift_pp", "wilson_ci_low",
                       "wilson_ci_high", "odds_ratio", "q_value", "adjusted_odds_ratio",
                       "adjusted_ci_low", "adjusted_ci_high", "adjusted_q_value",
                       "mean_abs_shap", "shap_rank", "evidence_category"]
        maximum = int(frame.support_n.dropna().max()) if frame.support_n.notna().any() else 1
        with st.container(horizontal=True, vertical_alignment="bottom"):
            minimum = st.number_input(
                "Minimum support", min_value=1, max_value=maximum,
                value=min(200, maximum), step=25, key=f"evidence_support_{feature_type}",
            )
            direction = st.segmented_control(
                "Association direction", ["All", "Positive", "Negative"], default="All",
                key=f"evidence_direction_{feature_type}",
            )
            significant = st.toggle("BH q < .05 only", key=f"evidence_q_{feature_type}")
        selected = frame.loc[frame.support_n.fillna(0) >= minimum].copy()
        if direction == "Positive":
            selected = selected.loc[selected.uplift_pp > 0]
        elif direction == "Negative":
            selected = selected.loc[selected.uplift_pp < 0]
        if significant:
            selected = selected.loc[selected.q_value < .05]
        if "evidence_category" in selected:
            categories = sorted(selected.evidence_category.dropna().unique())
            chosen = st.multiselect("Evidence classification", categories, default=categories,
                                    key=f"evidence_category_{feature_type}")
            selected = selected.loc[selected.evidence_category.isin(chosen)]
        st.dataframe(selected[[column for column in columns if column in selected]],
                     hide_index=True, key=f"evidence_table_{feature_type}")

st.subheader("Evidence comparison")
options = evidence.loc[evidence.uplift_pp.notna(), ["feature", "feature_type"]].copy()
options["label"] = options.feature + " (" + options.feature_type + ")"
choice = st.selectbox("Feature", options.label.sort_values(), key="evidence_comparison_feature")
row = evidence.loc[options.index[options.label == choice][0]]
with st.container(horizontal=True):
    with st.container(border=True):
        st.markdown("**Raw association**")
        st.metric("Top-four rate", "Unavailable" if pd.isna(row.top4_rate) else f"{row.top4_rate:.1%}")
        st.metric("Uplift", "Unavailable" if pd.isna(row.uplift_pp) else f"{row.uplift_pp:+.1f} pp")
        st.caption("Unadjusted cohort association")
    with st.container(border=True):
        st.markdown("**Adjusted association**")
        st.metric("Adjusted OR", "Not directly mapped" if pd.isna(row.adjusted_odds_ratio)
                  else f"{row.adjusted_odds_ratio:.2f}")
        if pd.notna(row.adjusted_ci_low):
            st.caption(f"95% CI {row.adjusted_ci_low:.2f}–{row.adjusted_ci_high:.2f}")
        else:
            st.caption("Null means no defensible feature-level mapping")
    with st.container(border=True):
        st.markdown("**Predictive importance**")
        st.metric("Mean |SHAP|", "Not directly mapped" if pd.isna(row.mean_abs_shap)
                  else f"{row.mean_abs_shap:.4f}")
        st.caption("Frozen XGBoost validation predictions")

st.info("These measures answer different questions and should not be interpreted as interchangeable.")


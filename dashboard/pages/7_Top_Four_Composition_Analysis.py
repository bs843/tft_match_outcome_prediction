"""Read-only exploration of end-state composition associations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.formatting import artifact_warning, method_note, section_kicker
from dashboard.components.loaders import load_composition_manifest, load_composition_table
from dashboard.components.validation import DashboardDataError


section_kicker("Post-model descriptive analysis · Patch 17.8")
st.title("Top-four composition analysis")
method_note(
    "These end-state patterns are associative, not causal. Final boards reflect prior economy, "
    "survival, item access and player decisions. Filters change presentation only and never "
    "retrain, tune or select a model."
)

try:
    baseline = load_composition_table("global_baseline.csv").iloc[0]
    manifest = load_composition_manifest()
    with st.container(horizontal=True):
        st.metric("Participant boards", f"{int(baseline['participant_observations']):,}", border=True)
        st.metric("Matches", f"{int(baseline['unique_matches']):,}", border=True)
        st.metric("Top-four baseline", f"{baseline['top_four_rate']:.1%}", border=True)
        st.metric("Average placement", f"{baseline['average_placement']:.2f}", border=True)
except (DashboardDataError, KeyError) as error:
    artifact_warning(error, "Composition-analysis baseline")
    st.stop()

view = st.segmented_control(
    "Analysis view",
    ["Feature associations", "Interactions", "Rank tiers", "Model importance"],
    default="Feature associations",
    key="composition_view",
)

if view == "Feature associations":
    feature_type = st.segmented_control(
        "Feature type", ["Units", "Items", "Traits", "Unit star levels"],
        default="Units", key="composition_feature_type",
    )
    sources = {
        "Units": ("unit_top4_analysis.csv", "unit"),
        "Items": ("item_top4_analysis.csv", "item"),
        "Traits": ("trait_top4_analysis.csv", "trait_threshold"),
        "Unit star levels": ("unit_star_top4_analysis.csv", "unit"),
    }
    table_name, label_column = sources[feature_type]
    table = load_composition_table(table_name)
    threshold_key = {"Units": "units", "Items": "items", "Traits": "traits",
                     "Unit star levels": "unit_star"}[feature_type]
    with st.container(horizontal=True, vertical_alignment="bottom"):
        minimum = st.number_input(
            "Minimum appearances", min_value=1, max_value=int(table["appearances"].max()),
            value=int(manifest["thresholds"][threshold_key]), step=25,
            key="composition_minimum",
        )
        query = st.text_input("Search features", placeholder="Type a unit, item or trait",
                              key="composition_search")
        star_values = table.get("star_level", pd.Series(dtype=int)).dropna().unique()
        star = st.selectbox("Star level", ["All", *map(str, sorted(star_values))],
                            key="composition_star")
    filtered = table.loc[table["appearances"] >= minimum].copy()
    if query:
        filtered = filtered.loc[filtered[label_column].str.contains(query, case=False, na=False)]
    if star != "All" and "star_level" in filtered:
        filtered = filtered.loc[filtered["star_level"].astype(str) == star]
    filtered = filtered.sort_values("top_four_uplift_pp", ascending=False)
    chart = filtered.head(20).sort_values("top_four_uplift_pp")
    st.bar_chart(chart, x="top_four_uplift_pp", y=label_column, horizontal=True,
                 x_label="Uplift versus 50% baseline (percentage points)", y_label=feature_type)
    columns = [label_column, "star_level", "appearances", "pick_rate", "top_four_rate",
               "top_four_uplift_pp", "average_placement", "top_four_rate_ci95_low",
               "top_four_rate_ci95_high", "odds_ratio", "fdr_bh_q_value"]
    display = filtered[[column for column in columns if column in filtered]].copy()
    st.dataframe(
        display, hide_index=True,
        column_config={
            "pick_rate": st.column_config.NumberColumn("Pick rate", format="percent"),
            "top_four_rate": st.column_config.NumberColumn("Top-four rate", format="percent"),
            "top_four_uplift_pp": st.column_config.NumberColumn("Uplift (pp)", format="%.2f"),
            "average_placement": st.column_config.NumberColumn("Average placement", format="%.2f"),
        },
        key="composition_feature_table",
    )
    st.caption(f"{len(filtered):,} rows shown after presentation-only filters.")

elif view == "Interactions":
    interactions = load_composition_table("feature_interactions.csv")
    interaction_type = st.selectbox("Interaction type", sorted(interactions["interaction_type"].unique()),
                                    key="composition_interaction_type")
    minimum = st.number_input("Minimum interaction appearances", min_value=1,
                              value=int(manifest["thresholds"]["interactions"]), step=25,
                              key="composition_interaction_minimum")
    selected = interactions.loc[(interactions["interaction_type"] == interaction_type) &
                                (interactions["appearances"] >= minimum)].copy()
    options = ["All", *sorted(set(selected["feature_a"]) | set(selected["feature_b"]))]
    feature = st.selectbox("Unit, item or trait", options, key="composition_interaction_feature")
    if feature != "All":
        selected = selected.loc[(selected["feature_a"] == feature) | (selected["feature_b"] == feature)]
    st.dataframe(selected.sort_values("uplift_vs_global_pp", ascending=False), hide_index=True,
                 key="composition_interaction_table")
    st.caption("All combinations are exploratory and satisfy the displayed minimum support.")

elif view == "Rank tiers":
    rank = load_composition_table("rank_stratified_feature_analysis.csv")
    feature_type = st.selectbox("Feature type", sorted(rank["feature_type"].unique()),
                                key="composition_rank_feature_type")
    tiers = sorted(rank["rank_tier"].dropna().unique())
    selected_tiers = st.multiselect("Rank tiers", tiers, default=tiers[:5],
                                    key="composition_rank_tiers")
    features = sorted(rank.loc[rank["feature_type"] == feature_type, "feature"].unique())
    feature = st.selectbox("Unit, item or trait", ["All", *features], key="composition_rank_feature")
    selected = rank.loc[(rank["feature_type"] == feature_type) &
                        rank["rank_tier"].isin(selected_tiers)].copy()
    if feature != "All": selected = selected.loc[selected["feature"] == feature]
    selected = selected.loc[selected["appearances"] >= int(manifest["thresholds"]["rank_cells"])]
    st.dataframe(selected.sort_values(["feature", "rank_tier"]), hide_index=True,
                 key="composition_rank_table")
    st.caption("Retrieval-time rank is descriptive metadata, not historical match-time rank.")

else:
    shap = load_composition_table("shap_feature_importance.csv")
    family = st.selectbox("Feature family", ["All", *sorted(shap["feature_family"].unique())],
                          key="composition_shap_family")
    top_n = st.select_slider("Number of features", options=[10, 15, 20, 30], value=20,
                             key="composition_shap_top_n")
    selected = shap if family == "All" else shap.loc[shap["feature_family"] == family]
    selected = selected.head(top_n)
    st.bar_chart(selected.sort_values("mean_absolute_shap"), x="mean_absolute_shap", y="feature",
                 horizontal=True, x_label="Mean absolute TreeSHAP", y_label="Feature")
    st.dataframe(selected, hide_index=True, key="composition_shap_table")
    st.caption(
        "Native TreeSHAP values describe the frozen development XGBoost on validation data. "
        "They explain model behaviour, not causal importance."
    )

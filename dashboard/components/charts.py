"""Presentation-only transformations for dashboard charts and tables."""

from __future__ import annotations

import re

import pandas as pd


FEATURE_LABELS = {
    "level": "Player level",
    "composition_size": "Composition size",
    "structure__unique_units": "Unique units",
    "structure__mean_star": "Mean unit star level",
    "structure__max_star": "Maximum star level",
    "structure__two_star_units": "Two-star units",
    "structure__three_star_units": "Three-star units",
    "structure__total_items": "Total equipped items",
    "structure__unique_items": "Unique item types",
    "structure__active_traits": "Active traits",
    "structure__trait_strength_total": "Total trait strength",
}


def humanize_feature(feature: str) -> str:
    """Create a concise display label without changing analytical meaning."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    value = feature
    for prefix, label in (
        ("unit_presence__", "Unit present · "),
        ("unit_star__", "Unit star level · "),
        ("trait_presence__", "Trait present · "),
        ("trait_strength__", "Trait strength · "),
        ("item_count__", "Item count · "),
    ):
        if value.startswith(prefix):
            value = label + value.removeprefix(prefix)
            break
    value = re.sub(r"^tft\d*_?", "", value, flags=re.IGNORECASE)
    return value.replace("_", " ").strip().title()


def feature_family_interpretation(frame: pd.DataFrame) -> list[str]:
    """Produce neutral, non-causal descriptions from the frozen progression."""
    ordered = frame.sort_values("feature_family").reset_index(drop=True)
    messages: list[str] = []
    for index in range(1, len(ordered)):
        previous = ordered.iloc[index - 1]
        current = ordered.iloc[index]
        change = float(current["f1"] - previous["f1"])
        addition = {
            "F2": "unit and star-level information",
            "F3": "trait information",
            "F4": "item information",
            "F5": "composition-structure summaries",
        }.get(str(current["feature_family"]), "the next cumulative feature family")
        qualifier = "higher" if change > 0 else "lower"
        messages.append(
            f"Adding {addition} was associated with {qualifier} validation F1 "
            f"({change:+.3f})."
        )
    if len(ordered) >= 2:
        last_change = float(ordered.iloc[-1]["f1"] - ordered.iloc[-2]["f1"])
        if abs(last_change) < 0.005:
            messages.append(
                "The final increment from F4 to F5 was marginal in the frozen "
                f"logistic family comparison ({last_change:+.3f} F1)."
            )
    return messages


def outcome_comparison(frame: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Return mean/median display data for one approved aggregate variable."""
    selected = frame.loc[frame["variable"] == variable, ["outcome", "mean", "median"]]
    return selected.set_index("outcome").rename(
        index={"bottom_four": "Bottom four", "top_four": "Top four"},
        columns={"mean": "Mean", "median": "Median"},
    )


def format_metric_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Select and consistently round display metrics."""
    output = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    numeric = output.select_dtypes(include="number").columns
    output[numeric] = output[numeric].round(4)
    return output


def add_feature_labels(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["display_feature"] = output["feature"].map(humanize_feature)
    return output


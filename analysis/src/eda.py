from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def run_eda(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Write reproducible descriptive tables and dependency-free SVG figures."""
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    placement = (frame.groupby("placement", as_index=False)
                 .agg(participants=("participant_hash", "size"),
                      mean_level=("level", "mean"),
                      mean_composition_size=("composition_size", "mean")))
    placement.to_csv(tables / "placement_summary.csv", index=False)

    rank = (frame.groupby(["rank_status", "rank_tier", "rank_division"], dropna=False,
                          as_index=False)
            .agg(participants=("participant_hash", "size"),
                 top4_rate=("top4", "mean"), mean_placement=("placement", "mean")))
    rank.to_csv(tables / "rank_descriptive_summary.csv", index=False)
    regional = (frame.groupby("platform_region", as_index=False)
                .agg(matches=("match_hash", "nunique"),
                     participants=("participant_hash", "size"),
                     top4_rate=("top4", "mean"),
                     mean_placement=("placement", "mean")))
    regional.to_csv(tables / "regional_distribution.csv", index=False)
    (frame.groupby(["platform_region", "placement"], as_index=False)
     .agg(participants=("participant_hash", "size"))
     .to_csv(tables / "placement_distribution_by_region.csv", index=False))
    (frame.groupby(["platform_region", "rank_status", "rank_tier", "rank_division"],
                   dropna=False, as_index=False)
     .agg(participants=("participant_hash", "size"))
     .to_csv(tables / "rank_distribution_by_region.csv", index=False))

    board = pd.DataFrame(_board_frequency_rows(frame, "top4"))
    board.to_csv(tables / "board_component_frequency_top4_bottom4.csv", index=False)
    by_placement = pd.DataFrame(_board_frequency_rows(frame, "placement"))
    by_placement.to_csv(tables / "board_component_frequency_by_placement.csv", index=False)
    top_bottom = (frame.assign(outcome_group=frame["top4"].map({1: "top4", 0: "bottom4"}))
                  .groupby("outcome_group", as_index=False)
                  .agg(participants=("participant_hash", "size"),
                       mean_level=("level", "mean"),
                       mean_composition_size=("composition_size", "mean")))
    top_bottom.to_csv(tables / "top4_bottom4_summary.csv", index=False)
    integrity = pd.DataFrame([{
        "matches": int(frame["match_hash"].nunique()),
        "participants": int(len(frame)),
        "duplicate_match_participants": int(frame[["match_hash", "participant_hash"]].duplicated().sum()),
        "missing_rank_status": int(frame["rank_status"].isna().sum()),
        "placement_min": int(frame["placement"].min()),
        "placement_max": int(frame["placement"].max()),
    }])
    integrity.to_csv(tables / "dataset_integrity.csv", index=False)
    frame[["placement", "top4", "level", "composition_size"]].describe().transpose().to_csv(
        tables / "modelling_dataset_summary.csv"
    )

    _bar_svg(figures / "placement_distribution.svg", "Participant placement distribution",
             [str(x) for x in placement["placement"]], placement["participants"].tolist())
    _bar_svg(figures / "mean_level_by_placement.svg", "Mean level by placement",
             [str(x) for x in placement["placement"]], placement["mean_level"].tolist())
    _bar_svg(figures / "top4_bottom4_level.svg", "Mean level: top four vs bottom four",
             top_bottom["outcome_group"].tolist(), top_bottom["mean_level"].tolist())
    return {
        "tables": sorted(path.name for path in tables.glob("*.csv")),
        "figures": sorted(path.name for path in figures.glob("*.svg")),
    }


def _board_frequency_rows(frame: pd.DataFrame, group_column: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for column, component_type, key in (
        ("units_json", "unit", "name"), ("traits_json", "trait", "name"),
        ("items_json", "item", None),
    ):
        for group_value, subset in frame.groupby(group_column):
            counts: Counter[str] = Counter()
            for value in subset[column]:
                entries = json.loads(value)
                counts.update(str(entry[key]) if key else str(entry) for entry in entries)
            denominator = max(1, len(subset))
            for name, count in counts.items():
                label = ("top4" if int(group_value) else "bottom4") if group_column == "top4" else str(group_value)
                result.append({"component_type": component_type, "component": name,
                               "grouping": group_column, "group": label,
                               "count": count, "per_participant": count / denominator})
    return sorted(result, key=lambda row: (row["component_type"], row["component"], row["group"]))


def _bar_svg(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    width, height, margin = 760, 440, 60
    chart_width, chart_height = width - 2 * margin, height - 2 * margin
    maximum = max([float(v) for v in values] or [1.0]) or 1.0
    bar_width = chart_width / max(1, len(values)) * 0.72
    gap = chart_width / max(1, len(values))
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
                f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>']
    for index, (label, raw_value) in enumerate(zip(labels, values)):
        value = float(raw_value)
        bar_height = chart_height * value / maximum
        x = margin + index * gap + (gap - bar_width) / 2
        y = height - margin - bar_height
        elements.extend([
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#4063d8"/>',
            f'<text x="{x+bar_width/2:.1f}" y="{height-margin+18}" text-anchor="middle" font-family="sans-serif" font-size="11">{html.escape(str(label))}</text>',
            f'<text x="{x+bar_width/2:.1f}" y="{max(45, y-5):.1f}" text-anchor="middle" font-family="sans-serif" font-size="10">{value:.2f}</text>',
        ])
    elements.append("</svg>\n")
    path.write_text("\n".join(elements), encoding="utf-8")

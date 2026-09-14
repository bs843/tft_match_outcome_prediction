from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .common import write_table
from .verify import VerifiedInputs


def build_descriptive_outputs(verified: VerifiedInputs, output: Path, logger) -> dict[str, Any]:
    eda = output / "eda"
    tables = output / "tables"
    eda.mkdir(parents=True, exist_ok=True)
    participant_parts = []
    feature_parts = []
    for split in ("train", "validation", "test"):
        participant_parts.append(pd.read_csv(
            verified.production_root / "data" / f"{split}_participants.csv.gz"
        ))
        feature_parts.append(pd.read_csv(
            verified.production_root / "data" / "features" / "F5" / f"{split}_X.csv.gz"
        ))
    participants = pd.concat(participant_parts, ignore_index=True)
    features = pd.concat(feature_parts, ignore_index=True)
    if len(participants) != 40000 or len(features) != 40000:
        raise RuntimeError("Full frozen descriptive cohort does not reconcile to 40,000 rows")

    placement = participants.groupby("placement", as_index=False).agg(
        participants=("participant_hash", "size"), matches=("match_hash", "nunique")
    )
    placement["proportion"] = placement["participants"] / len(participants)
    write_table(placement, eda / "placement_distribution.csv")
    _bar(placement["placement"].astype(str), placement["participants"],
         eda / "placement_distribution.png", "Final placement", "Participants",
         "Frozen cohort placement distribution")
    class_distribution = (participants.assign(
        outcome=np.where(participants["top4"].astype(int).eq(1), "top_four", "bottom_four")
    ).groupby("outcome", as_index=False).agg(participants=("participant_hash", "size"),
                                              matches=("match_hash", "nunique")))
    class_distribution["proportion"] = class_distribution["participants"] / len(participants)
    write_table(class_distribution, eda / "class_distribution.csv")
    _bar(class_distribution["outcome"], class_distribution["participants"],
         eda / "class_distribution.png", "Outcome", "Participants",
         "Top-four versus bottom-four class balance")

    outcome = np.where(participants["top4"].astype(int).eq(1), "top_four", "bottom_four")
    numeric_columns = ["level", "composition_size"] + [
        column for column in features.columns if column.startswith("structure__")
    ]
    numeric = pd.concat([
        participants[["level", "composition_size"]].reset_index(drop=True),
        features[[column for column in numeric_columns if column.startswith("structure__")]],
    ], axis=1)
    numeric["outcome"] = outcome
    numeric_summary = (numeric.groupby("outcome")[numeric_columns]
                       .agg(["count", "mean", "std", "median", "min", "max"]))
    numeric_summary.columns = [f"{column}_{stat}" for column, stat in numeric_summary.columns]
    numeric_summary = numeric_summary.reset_index()
    write_table(numeric_summary, eda / "top4_bottom4_numeric_summary.csv")

    unit = _feature_outcome_summary(features, outcome, "unit_presence__", "unit")
    trait = _feature_outcome_summary(features, outcome, "trait_presence__", "trait")
    item = _feature_outcome_summary(features, outcome, "item_count__", "item")
    write_table(unit, eda / "unit_frequency_by_outcome.csv")
    write_table(trait, eda / "trait_frequency_by_outcome.csv")
    write_table(item, eda / "item_frequency_by_outcome.csv")
    structure = _long_numeric_summary(numeric, numeric_columns)
    write_table(structure, eda / "composition_structure_by_outcome.csv")

    rank = (participants.assign(
        rank_stratum=participants["rank_tier"].fillna("UNKNOWN").astype(str).str.upper()
    ).groupby("rank_stratum", as_index=False).agg(
        participants=("participant_hash", "size"), matches=("match_hash", "nunique"),
        top_four_rate=("top4", "mean"), mean_placement=("placement", "mean")))
    region = participants.groupby("platform_region", as_index=False).agg(
        participants=("participant_hash", "size"), matches=("match_hash", "nunique"),
        top_four_rate=("top4", "mean"), mean_placement=("placement", "mean"))
    write_table(rank, eda / "rank_distribution.csv")
    write_table(region, eda / "region_distribution.csv")

    tests = _prespecified_tests(numeric, [
        "level", "composition_size", "structure__unique_units", "structure__mean_star",
        "structure__total_items", "structure__active_traits",
    ])
    write_table(tests, tables / "prespecified_group_tests.csv",
                tables / "prespecified_group_tests.md")
    dataset_summary = pd.DataFrame([
        {"measure": "accepted_matches", "value": 5000},
        {"measure": "participant_rows", "value": len(participants)},
        {"measure": "patch", "value": "17.8"},
        {"measure": "train_matches", "value": 3500},
        {"measure": "validation_matches", "value": 750},
        {"measure": "test_matches", "value": 750},
        {"measure": "source_predictors", "value": verified.source_predictor_count},
        {"measure": "encoded_f5_columns", "value": verified.encoded_feature_count},
        {"measure": "top_four_rows", "value": int(participants["top4"].astype(int).sum())},
        {"measure": "bottom_four_rows", "value": int((1-participants["top4"].astype(int)).sum())},
    ])
    write_table(dataset_summary, tables / "dataset_summary.csv", tables / "dataset_summary.md")
    logger.info("Generated descriptive Aim-1 outputs for %d frozen participants", len(participants))
    return {"participants": participants, "features": features, "dataset": dataset_summary,
            "rank": rank, "region": region, "tests": tests}


def _feature_outcome_summary(features: pd.DataFrame, outcome: np.ndarray,
                             prefix: str, label: str) -> pd.DataFrame:
    columns = [column for column in features if column.startswith(prefix)]
    frame = features[columns].copy()
    frame["outcome"] = outcome
    grouped = frame.groupby("outcome")[columns].mean().transpose().reset_index()
    grouped = grouped.rename(columns={"index": label})
    grouped[label] = grouped[label].str.removeprefix(prefix)
    grouped["top_four_minus_bottom_four"] = grouped.get("top_four", 0) - grouped.get("bottom_four", 0)
    return grouped.sort_values("top_four_minus_bottom_four", key=lambda x: x.abs(), ascending=False)


def _long_numeric_summary(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for outcome, subset in frame.groupby("outcome"):
        for column in columns:
            rows.append({"outcome": outcome, "variable": column, "n": len(subset),
                         "mean": subset[column].mean(), "sd": subset[column].std(),
                         "median": subset[column].median(), "q1": subset[column].quantile(.25),
                         "q3": subset[column].quantile(.75)})
    return pd.DataFrame(rows)


def _prespecified_tests(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    top = frame[frame["outcome"] == "top_four"]
    bottom = frame[frame["outcome"] == "bottom_four"]
    for column in columns:
        x = top[column].to_numpy()
        y = bottom[column].to_numpy()
        statistic, p_value = mannwhitneyu(x, y, alternative="two-sided")
        rank_biserial = 2 * statistic / (len(x) * len(y)) - 1
        rows.append({"variable": column, "test": "Mann-Whitney U",
                     "top_four_n": len(x), "bottom_four_n": len(y),
                     "top_four_median": np.median(x), "bottom_four_median": np.median(y),
                     "statistic": statistic, "p_value": p_value,
                     "rank_biserial_correlation": rank_biserial,
                     "interpretation_scope": "pre-specified statistical association; not causality"})
    result = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    count = len(result)
    adjusted = np.maximum.accumulate(np.minimum(1, result["p_value"].to_numpy() *
                                                np.arange(count, 0, -1)))
    result["holm_adjusted_p"] = adjusted
    return result.sort_values("variable").reset_index(drop=True)


def _bar(labels, values, path: Path, xlabel: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color="#4C72B0")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)

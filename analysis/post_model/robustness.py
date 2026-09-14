from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import safe_binary_metrics, write_table


RANK_ORDER = ["GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "UNRANKED", "UNKNOWN"]


def build_robustness_outputs(predictions: pd.DataFrame, output: Path, logger) -> dict[str, pd.DataFrame]:
    tables = output / "tables"
    figures = output / "figures"
    rank = predictions.copy()
    rank["rank_stratum"] = rank["rank_tier"].fillna("UNKNOWN").astype(str).str.upper()
    rank.loc[rank["rank_status"].astype(str).str.lower().eq("unranked"), "rank_stratum"] = "UNRANKED"
    rank_metrics = _stratified_metrics(rank, "rank_stratum")
    rank_metrics["order"] = rank_metrics["rank_stratum"].map(
        {value: index for index, value in enumerate(RANK_ORDER)}
    ).fillna(len(RANK_ORDER))
    rank_metrics = rank_metrics.sort_values(["order", "rank_stratum"]).drop(columns="order")
    write_table(rank_metrics, tables / "test_metrics_by_rank.csv",
                tables / "test_metrics_by_rank.md")
    _strata_figure(rank_metrics, "rank_stratum", figures / "test_performance_by_rank.png",
                   "Final-test performance by rank stratum")

    region = predictions.copy()
    region["region_stratum"] = region["platform_region"].fillna("UNKNOWN").astype(str)
    region_metrics = _stratified_metrics(region, "region_stratum").sort_values("region_stratum")
    write_table(region_metrics, tables / "test_metrics_by_region.csv",
                tables / "test_metrics_by_region.md")
    _strata_figure(region_metrics, "region_stratum",
                   figures / "test_performance_by_region.png",
                   "Final-test performance by platform region")
    logger.info("Calculated evaluation-only robustness for %d rank and %d region strata",
                len(rank_metrics), len(region_metrics))
    return {"rank": rank_metrics, "region": region_metrics}


def _stratified_metrics(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group, subset in frame.groupby(group_column, dropna=False, sort=True):
        metrics = safe_binary_metrics(subset["top4"].astype(int),
                                      subset["prediction"].astype(int), subset["probability"])
        rows.append({
            group_column: group, "n": len(subset),
            "matches": subset["match_hash"].nunique(),
            "prevalence": float(subset["top4"].mean()),
            **metrics,
            "roc_auc_defined": metrics["roc_auc"] is not None,
            "pr_auc_defined": metrics["pr_auc"] is not None,
        })
    return pd.DataFrame(rows)


def _strata_figure(frame: pd.DataFrame, label: str, path: Path, title: str) -> None:
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(max(8, len(frame) * 1.1), 5.5))
    width = 0.26
    for offset, metric, name in ((-width, "f1", "F1"), (0, "roc_auc", "ROC-AUC"),
                                 (width, "pr_auc", "PR-AUC")):
        ax.bar(x + offset, frame[metric].astype(float), width, label=name)
    ax.set_xticks(x, [f"{name}\n(n={n})" for name, n in zip(frame[label], frame["n"])],
                  rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)

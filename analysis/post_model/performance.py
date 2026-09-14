from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import dataframe_markdown, write_json, write_table, write_text
from .verify import VerifiedInputs


CLASSIFICATION_MODELS = [
    "dummy", "logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"
]


def build_performance_outputs(verified: VerifiedInputs, output: Path,
                              logger) -> dict[str, Any]:
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    source = verified.production_root / "modelling" / "tables"
    cv = pd.read_csv(source / "cross_validation_metrics.csv")
    validation = pd.read_csv(source / "validation_metrics.csv")
    final = pd.read_csv(source / "final_test_metrics.csv")
    eligible = validation[
        (validation["task"] == "top_four_classification")
        & validation["selection_eligible"].astype(str).str.lower().eq("true")
    ].copy()
    cv_summary = (cv.groupby("model", as_index=False)["f1"]
                  .agg(cv_f1_mean="mean", cv_f1_std="std", cv_folds="count"))
    frame = pd.DataFrame({"model": CLASSIFICATION_MODELS}).merge(cv_summary, on="model", how="left")
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    validation_columns = eligible[["model", *metrics]].rename(
        columns={metric: f"validation_{metric}" for metric in metrics}
    )
    frame = frame.merge(validation_columns, on="model", how="left")
    final_top4 = final[final["target"] == "top4"]
    if len(final_top4) != 1:
        raise RuntimeError("Expected exactly one locked top-four final-test result")
    final_columns = final_top4[["model", *metrics]].rename(
        columns={metric: f"test_{metric}" for metric in metrics}
    )
    frame = frame.merge(final_columns, on="model", how="left")
    frame = calculate_f1_gaps(frame)
    frame["selected"] = frame["model"].eq(verified.selection["selected_model"])
    write_table(frame, tables / "model_performance_summary.csv",
                tables / "model_performance_summary.md")

    gap_columns = [
        "model", "cv_f1_mean", "cv_f1_std", "validation_f1", "test_f1",
        "cv_to_validation_f1_gap", "validation_to_test_f1_gap", "cv_to_test_f1_gap",
        "abs_cv_to_validation_f1_gap", "abs_validation_to_test_f1_gap",
        "abs_cv_to_test_f1_gap", "cv_to_test_gap_in_sd", "selected",
    ]
    gaps = frame[gap_columns].copy()
    write_table(gaps, tables / "generalisation_gaps.csv",
                tables / "generalisation_gaps.md")
    cv_out = cv_summary.sort_values("cv_f1_mean", ascending=False)
    write_table(cv_out, tables / "cv_model_summary.csv", tables / "cv_model_summary.md")
    write_table(eligible[["model", *metrics, "selected"]].sort_values("f1", ascending=False),
                tables / "validation_model_summary.csv",
                tables / "validation_model_summary.md")
    write_table(final, tables / "final_test_model_summary.csv",
                tables / "final_test_model_summary.md")
    write_table(cv, tables / "cv_fold_metrics.csv")

    ranked = validation_ranking(eligible)
    winner = ranked.iloc[0]
    runner_up = ranked.iloc[1]
    margin = float(winner["f1"] - runner_up["f1"])
    variability_reference = float(cv_summary.set_index("model").loc[
        [winner["model"], runner_up["model"]], "cv_f1_std"
    ].max())
    effectively_tied = margin <= variability_reference
    placement_rows = validation[
        (validation["task"] == "exact_placement")
        & validation["selection_eligible"].astype(str).str.lower().eq("true")
    ].sort_values("mae")
    audit = {
        "status": "PASS",
        "selection_excluded_test_metrics": True,
        "selection_locked_at_utc": verified.selection["locked_at_utc"],
        "test_first_accessed_at_utc": verified.manifest["test_accessed_at_utc"],
        "top_four": {
            "selected_model": winner["model"],
            "selection_metric": "f1",
            "selection_rule": verified.selection["selection_rule"],
            "cv_mean": float(cv_summary.set_index("model").loc[winner["model"], "cv_f1_mean"]),
            "cv_std": float(cv_summary.set_index("model").loc[winner["model"], "cv_f1_std"]),
            "validation_f1": float(winner["f1"]),
            "runner_up": runner_up["model"],
            "runner_up_validation_f1": float(runner_up["f1"]),
            "margin": margin,
            "effectively_tied_relative_to_observed_cv_sd": effectively_tied,
            "materiality_basis": (
                "Margin is compared with the larger observed five-fold CV standard deviation "
                "of the winner and runner-up; this is a descriptive stability comparison, "
                "not a significance threshold."
            ),
            "cv_variability_reference": variability_reference,
        },
        "exact_placement": {
            "selected_model": placement_rows.iloc[0]["model"],
            "selection_metric": "mae",
            "validation_mae": float(placement_rows.iloc[0]["mae"]),
            "runner_up": placement_rows.iloc[1]["model"],
            "runner_up_validation_mae": float(placement_rows.iloc[1]["mae"]),
            "margin": float(placement_rows.iloc[1]["mae"] - placement_rows.iloc[0]["mae"]),
            "selection_rule": verified.placement_selection["selection_rule"],
        },
    }
    write_json(output / "model_selection_audit.json", audit)
    write_text(output / "model_selection_audit.md", _audit_markdown(audit))
    _cv_variability_figure(cv_out, figures / "cv_model_variability.png")
    _validation_comparison_figure(frame, figures / "model_comparison_test_metrics.png")
    _generalisation_figure(frame, figures / "model_generalisation_comparison.png")
    logger.info("Consolidated CV, validation and final-test metrics for %d classifiers", len(frame))
    return {"performance": frame, "gaps": gaps, "audit": audit, "cv": cv,
            "validation": validation, "final": final}


def calculate_f1_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["cv_to_validation_f1_gap"] = result["validation_f1"] - result["cv_f1_mean"]
    result["validation_to_test_f1_gap"] = result["test_f1"] - result["validation_f1"]
    result["cv_to_test_f1_gap"] = result["test_f1"] - result["cv_f1_mean"]
    result["abs_cv_to_validation_f1_gap"] = result["cv_to_validation_f1_gap"].abs()
    result["abs_validation_to_test_f1_gap"] = result["validation_to_test_f1_gap"].abs()
    result["abs_cv_to_test_f1_gap"] = result["cv_to_test_f1_gap"].abs()
    result["cv_to_test_gap_in_sd"] = result["abs_cv_to_test_f1_gap"] / result["cv_f1_std"]
    return result


def validation_ranking(validation: pd.DataFrame) -> pd.DataFrame:
    """Rank exclusively by validation F1; test metrics are deliberately ignored."""
    return validation.sort_values("f1", ascending=False, kind="stable").reset_index(drop=True)


def _audit_markdown(audit: dict[str, Any]) -> str:
    top = audit["top_four"]
    placement = audit["exact_placement"]
    return f"""# Model-selection audit

Status: **{audit['status']}**

The top-four winner was locked at `{audit['selection_locked_at_utc']}` before the
test partition was first accessed at `{audit['test_first_accessed_at_utc']}`.
Final-test results were excluded from selection.

## Top-four classification

- Selected model: `{top['selected_model']}`
- Primary metric: validation F1
- CV F1: {top['cv_mean']:.3f} ± {top['cv_std']:.3f}
- Validation F1: {top['validation_f1']:.3f}
- Runner-up: `{top['runner_up']}` ({top['runner_up_validation_f1']:.3f})
- Margin: {top['margin']:.4f}
- Within observed CV variability: {top['effectively_tied_relative_to_observed_cv_sd']}

The materiality statement is descriptive: {top['materiality_basis']}

## Exact placement

- Selected model: `{placement['selected_model']}`
- Validation MAE: {placement['validation_mae']:.3f}
- Runner-up: `{placement['runner_up']}` ({placement['runner_up_validation_mae']:.3f})
- MAE improvement: {placement['margin']:.3f}
"""


def _cv_variability_figure(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(frame["model"], frame["cv_f1_mean"], yerr=frame["cv_f1_std"], capsize=4,
           color="#3B6FB6")
    ax.set_ylabel("Five-fold CV F1 (mean ± SD)")
    ax.set_title("Cross-validation stability by classifier")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _validation_comparison_figure(frame: pd.DataFrame, path: Path) -> None:
    plot = frame.set_index("model")
    metrics = ["validation_f1", "validation_roc_auc", "validation_pr_auc"]
    labels = ["Validation F1", "Validation ROC-AUC", "Validation PR-AUC"]
    x = np.arange(len(plot))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, (metric, label) in enumerate(zip(metrics, labels)):
        ax.bar(x + (index - 1) * width, plot[metric], width, label=label)
    selected = plot[plot["selected"]]
    ax.scatter([x[list(plot.index).index(selected.index[0])]], selected["test_f1"],
               marker="*", s=160, color="black", label="Selected model final-test F1", zorder=5)
    ax.set_xticks(x, plot.index, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Validation comparison; final-test overlay for locked winner only")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _generalisation_figure(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(frame))
    ax.plot(x, frame["cv_f1_mean"], "o", label="CV mean F1")
    ax.plot(x, frame["validation_f1"], "s", label="Validation F1")
    selected = frame["selected"]
    ax.plot(x[selected], frame.loc[selected, "test_f1"], "*", markersize=13,
            label="Final-test F1 (locked winner only)")
    ax.set_xticks(x, frame["model"], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("CV → validation → final-test generalisation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)

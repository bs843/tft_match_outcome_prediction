from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .common import feature_group, write_table
from .verify import VerifiedInputs


def build_interpretation_outputs(verified: VerifiedInputs, output: Path, logger,
                                 repeats: int = 5, shap_sample: int = 1000) -> dict[str, Any]:
    tables = output / "tables"
    figures = output / "figures"
    validation = pd.read_csv(verified.production_root / "data" /
                             "validation_participants.csv.gz")
    matrix = pd.read_csv(verified.production_root / "data" / "features" / "F5" /
                         "validation_X.csv.gz")
    development_model = joblib.load(
        verified.production_root / "modelling" / "models" / "development" /
        "top4_f5_xgboost.joblib"
    )
    result = permutation_importance(
        development_model, matrix, validation["top4"].astype(int), scoring="f1",
        n_repeats=repeats, random_state=4275, n_jobs=1,
    )
    importance = pd.DataFrame({
        "feature": matrix.columns,
        "feature_group": [feature_group(column) for column in matrix.columns],
        "mean_importance": result.importances_mean,
        "std_importance": result.importances_std,
        "repeats": repeats,
        "scoring_metric": "f1",
        "evaluation_partition": "validation",
        "model_artifact": "development/top4_f5_xgboost.joblib",
    }).sort_values("mean_importance", ascending=False)
    write_table(importance, tables / "permutation_importance.csv")
    _importance_figure(importance.head(20), figures / "permutation_importance_top20.png",
                       "Validation permutation importance — top 20 features",
                       "Mean decrease in F1")

    groups = (importance.assign(abs_mean=importance["mean_importance"].abs())
              .groupby("feature_group", as_index=False)
              .agg(feature_count=("feature", "size"),
                   signed_mean_importance_sum=("mean_importance", "sum"),
                   absolute_mean_importance_sum=("abs_mean", "sum"),
                   mean_absolute_importance=("abs_mean", "mean"))
              .sort_values("absolute_mean_importance_sum", ascending=False))
    groups["aggregation_rule"] = (
        "sum of absolute column-level mean permutation importance within deterministic prefix group"
    )
    write_table(groups, tables / "feature_group_importance.csv",
                tables / "feature_group_importance.md")
    _group_figure(groups, figures / "feature_group_importance.png")

    native = pd.DataFrame({
        "feature": matrix.columns,
        "feature_group": [feature_group(column) for column in matrix.columns],
        "native_importance": np.asarray(development_model.feature_importances_, dtype=float),
        "interpretation": "supplementary model-specific importance; not causal",
    }).sort_values("native_importance", ascending=False)
    write_table(native, tables / "native_xgboost_feature_importance.csv")
    warnings: list[str] = []
    shap_created = False
    try:
        import shap
        sample_n = min(shap_sample, len(matrix))
        sample = matrix.sample(n=sample_n, random_state=4275)
        explainer = shap.TreeExplainer(development_model)
        values = explainer.shap_values(sample)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        mean_abs = np.abs(values).mean(axis=0)
        shap_table = pd.DataFrame({
            "feature": matrix.columns,
            "feature_group": [feature_group(column) for column in matrix.columns],
            "mean_absolute_shap": mean_abs,
            "sample_size": sample_n,
            "seed": 4275,
            "partition": "validation",
        }).sort_values("mean_absolute_shap", ascending=False)
        write_table(shap_table, tables / "shap_feature_importance.csv")
        _shap_figure(shap_table.head(20), figures / "shap_summary.png")
        shap_created = True
    except Exception as exc:  # SHAP is explicitly optional.
        warning = f"SHAP unavailable or incompatible; skipped without affecting other outputs: {type(exc).__name__}: {exc}"
        warnings.append(warning)
        logger.warning(warning)
    logger.info("Computed permutation importance for %d encoded features using %d repeats",
                len(importance), repeats)
    return {"permutation": importance, "groups": groups, "native": native,
            "shap_created": shap_created, "warnings": warnings}


def _importance_figure(frame: pd.DataFrame, path: Path, title: str, xlabel: str) -> None:
    plot = frame.sort_values("mean_importance")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.barh(plot["feature"], plot["mean_importance"], xerr=plot["std_importance"],
            color="#4C72B0", capsize=2)
    ax.set(xlabel=xlabel, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _group_figure(frame: pd.DataFrame, path: Path) -> None:
    plot = frame.sort_values("absolute_mean_importance_sum")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(plot["feature_group"], plot["absolute_mean_importance_sum"], color="#55A868")
    ax.set(xlabel="Aggregated absolute mean permutation importance",
           title="Predictive contribution by frozen feature group")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _shap_figure(frame: pd.DataFrame, path: Path) -> None:
    plot = frame.sort_values("mean_absolute_shap")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.barh(plot["feature"], plot["mean_absolute_shap"], color="#C44E52")
    ax.set(xlabel="Mean absolute SHAP value",
           title="Validation SHAP magnitude — deterministic sample")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)

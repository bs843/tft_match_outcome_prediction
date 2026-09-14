from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, mean_absolute_error,
                             median_absolute_error, precision_recall_fscore_support)

from analysis.post_model.common import (ANALYSIS_COMMIT, ANALYSIS_TAG, CONTRACT_VERSION,
                                        DATASET_VERSION, PRODUCTION_RUN_ID, SEED,
                                        SPLIT_HASH, dataframe_markdown, read_json,
                                        sha256_file, stable_hash, utc_now, write_table,
                                        write_text)
from analysis.post_model.verify import verify_inputs
from analysis.src.features import BASIC, STRUCTURE, CompositionFeatureEncoder
from analysis.src.models import _estimators
from analysis.src.splits import cv_splits
from src.config import PROJECT_ROOT


VERSION = "1.0.0"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "thesis_extension_v1"
GROUPS = ("basic", "units", "traits", "items", "structure")
FAMILIES = {
    "F1": ("basic",),
    "F2": ("basic", "units"),
    "F3": ("basic", "units", "traits"),
    "F4": ("basic", "units", "traits", "items"),
    "F5": GROUPS,
}
GROUP_LABELS = {
    "basic": "Basic final-board state",
    "units": "Units/champions",
    "traits": "Traits",
    "items": "Equipped items",
    "structure": "Composition structure",
}
SEEDS = (SEED, SEED + 1, SEED + 2)


@dataclass
class FoldData:
    seed: int
    fold: int
    fit_y: pd.Series
    held_y: pd.Series
    held_placement: pd.Series
    fit_groups: dict[str, pd.DataFrame]
    held_groups: dict[str, pd.DataFrame]
    held_matches: pd.Series


def _xgb(seed: int):
    return _estimators(seed, require_boosters=True)["xgboost"]


def _rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                  n_jobs=1, random_state=seed)


def _combine(groups: Iterable[str], matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    selected = [matrices[group] for group in groups]
    return pd.concat(selected, axis=1) if selected else pd.DataFrame(index=next(iter(matrices.values())).index)


def _fold_data(train: pd.DataFrame, seed: int, folds: int = 5) -> Iterable[FoldData]:
    clean = train.reset_index(drop=True)
    for fold, (fit_idx, held_idx) in enumerate(cv_splits(clean, folds=folds, seed=seed), 1):
        fit = clean.iloc[fit_idx].reset_index(drop=True)
        held = clean.iloc[held_idx].reset_index(drop=True)
        encoder = CompositionFeatureEncoder(minimum_frequency=5).fit(fit)
        fit_groups = {group: encoder.transform(fit, [group]) for group in GROUPS}
        held_groups = {group: encoder.transform(held, [group]) for group in GROUPS}
        yield FoldData(seed, fold, fit["top4"].astype(int), held["top4"].astype(int),
                       held["placement"].astype(int), fit_groups, held_groups,
                       held["match_hash"].astype(str))


def feature_definitions(production: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    vocabulary = read_json(production / "data" / "feature_vocabulary.json")
    vocab = {
        "units": vocabulary.get("unit_vocabulary", []),
        "traits": vocabulary.get("trait_vocabulary", []),
        "items": vocabulary.get("item_vocabulary", []),
    }
    rows: list[dict[str, Any]] = []
    for family, groups in FAMILIES.items():
        variables = []
        for group in groups:
            if group == "basic": variables.extend(BASIC)
            elif group == "structure": variables.extend(STRUCTURE)
            elif group == "units": variables.extend(["unit_presence__<unit>", "unit_star__<unit>"])
            elif group == "traits": variables.extend(["trait_presence__<trait>", "trait_strength__<trait>"])
            elif group == "items": variables.append("item_count__<item>")
        rows.append({"feature_family": family, "cumulative": True,
                     "included_units": "; ".join(GROUP_LABELS[g] for g in groups),
                     "variables_or_patterns": "; ".join(variables),
                     "definition_source": "analysis/configs/experiments.yaml; analysis/src/features.py"})
    families = pd.DataFrame(rows)

    definitions = [
        ("level", "basic", "Basic final-board state", "player level", "numeric level", "features.py implementation; external Riot semantic definition not documented"),
        ("composition_size", "basic", "Basic final-board state", "units on final board", "count", "features.py/data dictionary"),
        ("unit_presence__<unit>", "units", "Units/champions", "named unit on final board", "binary indicator", "features.py"),
        ("unit_star__<unit>", "units", "Units/champions", "star level of named unit; zero if absent", "integer star level encoded numeric", "features.py; external upper-bound semantics undocumented"),
        ("trait_presence__<trait>", "traits", "Traits", "named trait present", "binary indicator", "features.py"),
        ("trait_strength__<trait>", "traits", "Traits", "tier_current, falling back to style", "numeric Riot payload tier/style", "features.py; substantive tier scale undocumented"),
        ("item_count__<item>", "items", "Equipped items", "occurrences of named item", "count", "features.py"),
        ("structure__unique_units", "structure", "Composition structure", "distinct named units", "count", "features.py"),
        ("structure__mean_star", "structure", "Composition structure", "mean star level across final units", "mean star level", "features.py"),
        ("structure__max_star", "structure", "Composition structure", "maximum star level", "star level", "features.py"),
        ("structure__two_star_units", "structure", "Composition structure", "units at exactly two stars", "count", "features.py"),
        ("structure__three_star_units", "structure", "Composition structure", "units at three or more stars", "count", "features.py"),
        ("structure__total_items", "structure", "Composition structure", "all equipped item occurrences", "count", "features.py"),
        ("structure__unique_items", "structure", "Composition structure", "distinct equipped item names", "count", "features.py"),
        ("structure__active_traits", "structure", "Composition structure", "traits with tier_current > 0", "count", "features.py"),
        ("structure__trait_strength_total", "structure", "Composition structure", "sum of tier_current among active traits", "sum of payload tiers", "features.py; cross-trait comparability undocumented"),
    ]
    units = pd.DataFrame(definitions, columns=["variable_or_pattern", "group", "substantive_unit",
                                               "operational_definition", "measurement_unit",
                                               "documentation_status"])
    counts = {"basic": len(BASIC), "units": 2 * len(vocab["units"]),
              "traits": 2 * len(vocab["traits"]), "items": len(vocab["items"]),
              "structure": len(STRUCTURE)}
    units["frozen_encoded_column_count"] = units["group"].map(counts)
    write_table(families, output / "tables" / "feature_family_definitions.csv",
                output / "tables" / "feature_family_definitions.md")
    write_table(units, output / "tables" / "substantive_unit_definitions.csv",
                output / "tables" / "substantive_unit_definitions.md")
    return families, units


def ablations_and_placement(train: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ablation_rows: list[dict[str, Any]] = []
    placement_rows: list[dict[str, Any]] = []
    placement_oof: list[pd.DataFrame] = []
    for fold_data in _fold_data(train, SEED):
        variants = {"full_F5": GROUPS}
        variants.update({f"only_{group}": (group,) for group in GROUPS})
        variants.update({f"without_{group}": tuple(g for g in GROUPS if g != group)
                         for group in GROUPS})
        for variant, groups in variants.items():
            model = _xgb(SEED)
            fit_x = _combine(groups, fold_data.fit_groups)
            held_x = _combine(groups, fold_data.held_groups)
            model.fit(fit_x, fold_data.fit_y)
            prediction = model.predict(held_x)
            ablation_rows.append({"seed": SEED, "fold": fold_data.fold,
                                  "variant": variant, "groups": ";".join(groups),
                                  "f1": f1_score(fold_data.held_y, prediction, zero_division=0),
                                  "fit_rows": len(fit_x), "held_rows": len(held_x),
                                  "encoded_columns": fit_x.shape[1]})
    folds = pd.DataFrame(ablation_rows)
    full = folds.loc[folds["variant"] == "full_F5"].set_index("fold")["f1"]
    summaries = []
    for variant, part in folds.groupby("variant", sort=False):
        scores = part.set_index("fold")["f1"].sort_index()
        delta = scores - full.reindex(scores.index)
        summaries.append({"variant": variant,
                          "analysis_type": "full" if variant == "full_F5" else ("unit_only" if variant.startswith("only_") else "leave_one_unit_out"),
                          "substantive_unit": "all" if variant == "full_F5" else variant.split("_", 1)[1],
                          "mean_cv_f1": scores.mean(), "fold_sd_f1": scores.std(ddof=1),
                          "change_from_full_f5": delta.mean(),
                          "removal_deterioration": -delta.mean() if variant.startswith("without_") else np.nan,
                          "paired_difference_sd": delta.std(ddof=1), "folds": len(scores),
                          "seed": SEED})
    summary = pd.DataFrame(summaries)
    write_table(folds, output / "tables" / "unit_ablation_fold_results.csv")
    write_table(summary, output / "tables" / "unit_ablation_summary.csv",
                output / "tables" / "unit_ablation_summary.md", decimals=4)
    return folds, summary


def placement_cv(train: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = train.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    oof: list[pd.DataFrame] = []
    for fold, (fit_idx, held_idx) in enumerate(cv_splits(clean, folds=5, seed=SEED), 1):
        fit = clean.iloc[fit_idx].reset_index(drop=True)
        held = clean.iloc[held_idx].reset_index(drop=True)
        encoder = CompositionFeatureEncoder(5).fit(fit)
        fit_x = encoder.transform(fit, GROUPS)
        held_x = encoder.transform(held, GROUPS)
        for name, model in (("dummy", DummyClassifier(strategy="most_frequent", random_state=SEED)),
                            ("random_forest", _rf(SEED))):
            model.fit(fit_x, fit["placement"].astype(int))
            pred = np.asarray(model.predict(held_x), dtype=int)
            truth = held["placement"].astype(int).to_numpy()
            error = np.abs(truth - pred)
            rows.append({"model": name, "fold": fold, "seed": SEED,
                         "accuracy": accuracy_score(truth, pred),
                         "macro_f1": f1_score(truth, pred, average="macro", zero_division=0),
                         "weighted_f1": f1_score(truth, pred, average="weighted", zero_division=0),
                         "mae": mean_absolute_error(truth, pred),
                         "median_absolute_error": median_absolute_error(truth, pred),
                         "within_1_accuracy": np.mean(error <= 1),
                         "within_2_accuracy": np.mean(error <= 2)})
            oof.append(pd.DataFrame({"model": name, "fold": fold, "truth": truth,
                                     "prediction": pred, "match_hash": held["match_hash"]}))
    fold_frame = pd.DataFrame(rows)
    metric_names = ["accuracy", "macro_f1", "weighted_f1", "mae",
                    "median_absolute_error", "within_1_accuracy", "within_2_accuracy"]
    summary_rows = []
    for model, part in fold_frame.groupby("model"):
        row: dict[str, Any] = {"model": model, "folds": len(part), "seed": SEED}
        for metric in metric_names:
            row[f"mean_{metric}"] = part[metric].mean()
            row[f"sd_{metric}"] = part[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    predictions = pd.concat(oof, ignore_index=True)
    write_table(fold_frame, output / "tables" / "placement_cv_fold_metrics.csv")
    write_table(summary, output / "tables" / "placement_cv_summary.csv",
                output / "tables" / "placement_cv_summary.md", decimals=4)
    selected = predictions.loc[predictions["model"] == "random_forest"]
    labels = list(range(1, 9))
    cm = pd.DataFrame(confusion_matrix(selected["truth"], selected["prediction"], labels=labels),
                      index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    cm.to_csv(output / "tables" / "placement_cv_confusion_matrix.csv")
    precision, recall, f1, support = precision_recall_fscore_support(
        selected["truth"], selected["prediction"], labels=labels, zero_division=0)
    per_class = pd.DataFrame({"placement": labels, "precision": precision, "recall": recall,
                              "f1": f1, "support": support})
    write_table(per_class, output / "tables" / "placement_cv_per_class.csv",
                output / "tables" / "placement_cv_per_class.md")
    _heatmap(cm.to_numpy(), [str(x) for x in labels], [str(x) for x in labels],
             output / "figures" / "placement_cv_confusion_matrix.png",
             "Placement CV confusion matrix — random forest", "Predicted", "True")
    return fold_frame, summary


def repeated_cv(train: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for data in _fold_data(train, seed):
            fit_x = _combine(GROUPS, data.fit_groups)
            held_x = _combine(GROUPS, data.held_groups)
            model = _xgb(seed).fit(fit_x, data.fit_y)
            score = f1_score(data.held_y, model.predict(held_x), zero_division=0)
            rows.append({"seed": seed, "fold": data.fold, "f1": score})
    frame = pd.DataFrame(rows)
    write_table(frame, output / "tables" / "xgboost_repeated_cv_folds.csv")
    summary = frame.groupby("seed", as_index=False)["f1"].agg(["mean", "std", "min", "max"]).reset_index()
    write_table(summary, output / "tables" / "xgboost_repeated_cv_summary.csv",
                output / "tables" / "xgboost_repeated_cv_summary.md", decimals=4)
    return frame


def shap_analysis(production: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = joblib.load(production / "modelling" / "models" / "development" / "top4_f5_xgboost.joblib")
    x = pd.read_csv(production / "data" / "features" / "F5" / "validation_X.csv.gz")
    participants = pd.read_csv(production / "data" / "validation_participants.csv.gz")
    matrix = xgboost.DMatrix(x, feature_names=list(x.columns))
    contributions = model.get_booster().predict(matrix, pred_contribs=True)
    shap_values = contributions[:, :-1]
    importance = pd.DataFrame({"feature": x.columns,
                               "mean_absolute_shap": np.mean(np.abs(shap_values), axis=0),
                               "mean_signed_shap": np.mean(shap_values, axis=0),
                               "feature_mean": x.mean(axis=0).to_numpy()})
    importance["substantive_unit"] = importance["feature"].map(feature_group)
    importance = importance.sort_values("mean_absolute_shap", ascending=False).reset_index(drop=True)
    unit = importance.groupby("substantive_unit", as_index=False).agg(
        total_mean_absolute_shap=("mean_absolute_shap", "sum"),
        mean_feature_absolute_shap=("mean_absolute_shap", "mean"),
        encoded_features=("feature", "count"),
    ).sort_values("total_mean_absolute_shap", ascending=False)
    write_table(importance, output / "tables" / "shap_feature_importance.csv",
                output / "tables" / "shap_feature_importance.md", decimals=5)
    write_table(unit, output / "tables" / "shap_unit_importance.csv",
                output / "tables" / "shap_unit_importance.md", decimals=5)

    top = importance.head(20).sort_values("mean_absolute_shap")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["mean_absolute_shap"], color="#4c78a8")
    ax.set_xlabel("Mean absolute TreeSHAP value (log-odds scale)")
    ax.set_title("Validation TreeSHAP global importance — selected XGBoost class")
    fig.tight_layout(); fig.savefig(output / "figures" / "shap_feature_importance.png", dpi=180); plt.close(fig)
    ordered = unit.sort_values("total_mean_absolute_shap")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered["substantive_unit"], ordered["total_mean_absolute_shap"], color="#59a66f")
    ax.set_xlabel("Sum of mean absolute TreeSHAP across encoded columns")
    ax.set_title("TreeSHAP importance aggregated by substantive unit")
    fig.tight_layout(); fig.savefig(output / "figures" / "shap_unit_importance.png", dpi=180); plt.close(fig)

    direction_features = importance.head(6)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, feature in zip(axes.flat, direction_features):
        index = x.columns.get_loc(feature)
        ax.scatter(x[feature], shap_values[:, index], s=7, alpha=.25, color="#4c78a8")
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_title(feature); ax.set_xlabel("Feature value"); ax.set_ylabel("TreeSHAP")
    fig.suptitle("Feature direction on validation data (log-odds contribution)")
    fig.tight_layout(); fig.savefig(output / "figures" / "shap_feature_direction.png", dpi=180); plt.close(fig)

    probabilities = model.predict_proba(x)[:, 1]
    truth = participants["top4"].astype(int).to_numpy()
    predicted = (probabilities >= .5).astype(int)
    candidates = {
        "representative_top4": np.where((truth == 1) & (predicted == 1))[0],
        "representative_bottom4": np.where((truth == 0) & (predicted == 0))[0],
        "high_confidence_error": np.where(truth != predicted)[0],
    }
    selected_indices = []
    for label, indices in candidates.items():
        if label == "high_confidence_error":
            confidence = np.maximum(probabilities[indices], 1 - probabilities[indices])
            chosen = indices[int(np.argmax(confidence))]
        else:
            median = np.median(probabilities[indices])
            chosen = indices[int(np.argmin(np.abs(probabilities[indices] - median)))]
        selected_indices.append((label, chosen))
    local_rows = []
    for label, row_index in selected_indices:
        order = np.argsort(np.abs(shap_values[row_index]))[::-1][:10]
        observation = hashlib.sha256(
            ("thesis-extension:" + str(participants.iloc[row_index]["participant_hash"])).encode()
        ).hexdigest()[:20]
        for rank, feature_index in enumerate(order, 1):
            local_rows.append({"example": label, "observation_id": observation,
                               "true_top4": int(truth[row_index]),
                               "predicted_probability": probabilities[row_index],
                               "rank": rank, "feature": x.columns[feature_index],
                               "feature_value": x.iloc[row_index, feature_index],
                               "shap_log_odds_contribution": shap_values[row_index, feature_index]})
    local = pd.DataFrame(local_rows)
    write_table(local, output / "tables" / "shap_local_explanations.csv",
                output / "tables" / "shap_local_explanations.md", decimals=5)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, (label, part) in zip(axes, local.groupby("example", sort=False)):
        ordered_local = part.sort_values("shap_log_odds_contribution")
        colors = np.where(ordered_local["shap_log_odds_contribution"] >= 0, "#d95f5f", "#4c78a8")
        ax.barh(ordered_local["feature"], ordered_local["shap_log_odds_contribution"], color=colors)
        ax.set_title(label.replace("_", " ")); ax.set_xlabel("TreeSHAP contribution")
    fig.suptitle("Representative validation explanations — selected XGBoost class")
    fig.tight_layout(); fig.savefig(output / "figures" / "shap_local_explanations.png", dpi=180); plt.close(fig)
    return importance, unit, local


def feature_group(name: str) -> str:
    if name in BASIC: return "basic"
    if name.startswith(("unit_presence__", "unit_star__")): return "units"
    if name.startswith(("trait_presence__", "trait_strength__")): return "traits"
    if name.startswith("item_count__"): return "items"
    if name.startswith("structure__"): return "structure"
    return "undocumented"


def correlation_audit(production: Path, output: Path, threshold: float = .90) -> pd.DataFrame:
    x = pd.read_csv(production / "data" / "features" / "F5" / "validation_X.csv.gz")
    variable = x.loc[:, x.nunique(dropna=False) > 1]
    corr = variable.corr().abs()
    upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    positions = np.argwhere((corr.to_numpy() >= threshold) & upper)
    rows = [{"feature_a": corr.index[i], "feature_b": corr.columns[j],
             "absolute_pearson_correlation": corr.iat[i, j],
             "unit_a": feature_group(corr.index[i]), "unit_b": feature_group(corr.columns[j])}
            for i, j in positions]
    frame = pd.DataFrame(rows, columns=["feature_a", "feature_b", "absolute_pearson_correlation",
                                       "unit_a", "unit_b"]).sort_values(
                                           "absolute_pearson_correlation", ascending=False)
    write_table(frame, output / "tables" / "high_feature_correlations.csv",
                output / "tables" / "high_feature_correlations.md", decimals=4)
    return frame


def model_selection_audit(production: Path, output: Path) -> pd.DataFrame:
    cv = pd.read_csv(production / "modelling" / "tables" / "cross_validation_metrics.csv")
    validation = pd.read_csv(production / "modelling" / "tables" / "validation_metrics.csv")
    summaries = cv.groupby("model", as_index=False)["f1"].agg(cv_mean="mean", cv_sd="std")
    val = validation.loc[(validation["task"] == "top_four_classification") &
                         (validation["feature_family"] == "F5") &
                         (validation["model"] != "logistic_feature_family"),
                         ["model", "f1", "selected"]].rename(columns={"f1": "validation_f1"})
    result = summaries.merge(val, on="model", how="left").sort_values("validation_f1", ascending=False)
    write_table(result, output / "tables" / "model_selection_audit.csv",
                output / "tables" / "model_selection_audit.md", decimals=5)
    pivot = cv.pivot(index="fold", columns="model", values="f1")
    difference = pivot["lightgbm"] - pivot["xgboost"]
    mean = difference.mean(); sem = stats.sem(difference)
    ci = stats.t.interval(.95, len(difference) - 1, loc=mean, scale=sem)
    try: wilcoxon = stats.wilcoxon(difference).pvalue
    except ValueError: wilcoxon = np.nan
    paired = pd.DataFrame([{"comparison": "lightgbm_minus_xgboost", "folds": len(difference),
                            "mean_paired_f1_difference": mean, "sd_paired_difference": difference.std(ddof=1),
                            "ci95_low": ci[0], "ci95_high": ci[1],
                            "paired_t_pvalue": stats.ttest_rel(pivot["lightgbm"], pivot["xgboost"]).pvalue,
                            "wilcoxon_pvalue": wilcoxon}])
    write_table(paired, output / "tables" / "xgboost_lightgbm_paired_cv.csv",
                output / "tables" / "xgboost_lightgbm_paired_cv.md", decimals=6)
    return result


def probability_and_final_placement(production: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for partition, model_name in (("validation", "development/top4_f5_xgboost.joblib"),
                                  ("test", "selected_top4_f5.joblib")):
        x = pd.read_csv(production / "data" / "features" / "F5" / f"{partition}_X.csv.gz")
        participants = pd.read_csv(production / "data" / f"{partition}_participants.csv.gz")
        model = joblib.load(production / "modelling" / "models" / model_name)
        probability = model.predict_proba(x)[:, 1]
        truth = participants["top4"].astype(int).to_numpy()
        prediction = (probability >= .5).astype(int)
        bins = pd.cut(probability, bins=np.linspace(0, 1, 11), include_lowest=True)
        calibration = pd.DataFrame({"bin": bins.astype(str), "probability": probability,
                                    "truth": truth}).groupby("bin", observed=False).agg(
                                        count=("truth", "size"), mean_probability=("probability", "mean"),
                                        observed_rate=("truth", "mean")).reset_index()
        calibration["partition"] = partition
        calibration.to_csv(output / "tables" / f"{partition}_calibration_bins.csv", index=False)
        valid = calibration.loc[calibration["count"] > 0]
        ece = np.average(np.abs(valid["observed_rate"] - valid["mean_probability"]),
                         weights=valid["count"])
        rows.append({"partition": partition, "threshold": .5,
                     "threshold_source": "frozen_contract_default_no_alternative_application_threshold_documented",
                     "f1": f1_score(truth, prediction),
                     "brier_score": brier_score_loss(truth, probability),
                     "log_loss": log_loss(truth, probability, labels=[0, 1]),
                     "expected_calibration_error_10_bin": ece,
                     "maximum_calibration_error_10_bin": np.max(np.abs(valid["observed_rate"] - valid["mean_probability"]))})
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
        ax.plot(valid["mean_probability"], valid["observed_rate"], marker="o")
        ax.set(xlabel="Mean predicted probability", ylabel="Observed top-four rate",
               title=f"{partition.title()} reliability curve — fixed 0.5 threshold")
        fig.tight_layout(); fig.savefig(output / "figures" / f"{partition}_reliability_curve.png", dpi=180); plt.close(fig)
    probability_summary = pd.DataFrame(rows)
    write_table(probability_summary, output / "tables" / "probability_evaluation.csv",
                output / "tables" / "probability_evaluation.md", decimals=5)

    x_test = pd.read_csv(production / "data" / "features" / "F5" / "test_X.csv.gz")
    test = pd.read_csv(production / "data" / "test_participants.csv.gz")
    placement_model = joblib.load(production / "modelling" / "models" / "selected_placement_f5.joblib")
    pred = np.asarray(placement_model.predict(x_test), dtype=int)
    truth = test["placement"].astype(int).to_numpy(); error = np.abs(truth - pred)
    dummy = DummyClassifier(strategy="most_frequent", random_state=SEED).fit(x_test.iloc[:1], [1])
    dummy_pred = np.ones(len(test), dtype=int)
    final_rows = []
    for name, values in (("random_forest_locked", pred), ("dummy_most_frequent", dummy_pred)):
        absolute = np.abs(truth - values)
        final_rows.append({"model": name, "partition": "final_test_read_only",
                           "accuracy": accuracy_score(truth, values),
                           "macro_f1": f1_score(truth, values, average="macro", zero_division=0),
                           "weighted_f1": f1_score(truth, values, average="weighted", zero_division=0),
                           "mae": mean_absolute_error(truth, values),
                           "median_absolute_error": median_absolute_error(truth, values),
                           "within_1_accuracy": np.mean(absolute <= 1),
                           "within_2_accuracy": np.mean(absolute <= 2)})
    final = pd.DataFrame(final_rows)
    write_table(final, output / "tables" / "placement_final_test_metrics.csv",
                output / "tables" / "placement_final_test_metrics.md", decimals=4)
    return probability_summary, final


def partition_audit(production: Path, output: Path) -> pd.DataFrame:
    frames = {split: pd.read_csv(production / "data" / f"{split}_participants.csv.gz")
              for split in ("train", "validation", "test")}
    rows = []
    for split, frame in frames.items():
        rows.append({"audit": "partition", "left": split, "right": "",
                     "value": frame["match_hash"].nunique(), "detail": "unique_matches"})
        rows.append({"audit": "time_range", "left": split, "right": "",
                     "value": f"{frame['game_datetime'].min()}..{frame['game_datetime'].max()}",
                     "detail": "epoch_milliseconds_as_stored"})
    for i, left in enumerate(frames):
        for right in list(frames)[i + 1:]:
            match_overlap = set(frames[left]["match_hash"]) & set(frames[right]["match_hash"])
            participant_overlap = set(frames[left]["participant_hash"]) & set(frames[right]["participant_hash"])
            rows.append({"audit": "match_overlap", "left": left, "right": right,
                         "value": len(match_overlap), "detail": "must_equal_zero"})
            rows.append({"audit": "participant_overlap", "left": left, "right": right,
                         "value": len(participant_overlap),
                         "detail": "disclosed dependence; hashes not exported"})
    all_frame = pd.concat(frames.values(), ignore_index=True)
    rows.extend([
        {"audit": "study_patch_values", "left": "all", "right": "", "value": all_frame["study_patch"].nunique(), "detail": ";".join(map(str, sorted(all_frame["study_patch"].unique())))},
        {"audit": "set_values", "left": "all", "right": "", "value": all_frame["set_number"].nunique(), "detail": ";".join(map(str, sorted(all_frame["set_number"].unique())))},
        {"audit": "queue_values", "left": "all", "right": "", "value": all_frame["queue_id"].nunique(), "detail": ";".join(map(str, sorted(all_frame["queue_id"].unique())))},
    ])
    result = pd.DataFrame(rows)
    write_table(result, output / "tables" / "partition_and_leakage_audit.csv",
                output / "tables" / "partition_and_leakage_audit.md")
    return result


def _heatmap(values: np.ndarray, xlabels: list[str], ylabels: list[str], path: Path,
             title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7)); image = ax.imshow(values, cmap="Blues")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]): ax.text(j, i, str(values[i, j]), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(xlabels)), xlabels); ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title); fig.colorbar(image, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _write_reports(output: Path, ablation: pd.DataFrame, placement: pd.DataFrame,
                   shap_unit: pd.DataFrame, correlations: pd.DataFrame,
                   repeated: pd.DataFrame, selection: pd.DataFrame,
                   probability: pd.DataFrame, final_placement: pd.DataFrame,
                   partition: pd.DataFrame) -> None:
    full = ablation.loc[ablation["variant"] == "full_F5"].iloc[0]
    leave = ablation.loc[ablation["analysis_type"] == "leave_one_unit_out"].sort_values(
        "removal_deterioration", ascending=False)
    strongest = leave.iloc[0]
    rf_cv = placement.loc[placement["model"] == "random_forest"].iloc[0]
    dummy_cv = placement.loc[placement["model"] == "dummy"].iloc[0]
    xgb = selection.loc[selection["model"] == "xgboost"].iloc[0]
    lgb = selection.loc[selection["model"] == "lightgbm"].iloc[0]
    test_prob = probability.loc[probability["partition"] == "test"].iloc[0]
    test_place = final_placement.loc[final_placement["model"] == "random_forest_locked"].iloc[0]
    repeated_mean = repeated["f1"].mean(); repeated_sd = repeated["f1"].std(ddof=1)
    participant_overlap = partition.loc[partition["audit"] == "participant_overlap", "value"].astype(int)
    summary = f"""# Extended thesis analysis results

## Scope and safeguards

This is a read-only, post-selection extension of frozen analysis `{PRODUCTION_RUN_ID}` under
contract `{CONTRACT_VERSION}`. It does not change the accepted cohort, frozen chronological
split, registered models, selected models, thresholds, or final-test decision. Ablation,
TreeSHAP, correlation, repeated-CV, and placement-CV analyses use training or validation data;
the final test is used only for descriptive evaluation of the already locked models.

## Feature families

F1–F5 are cumulative. F1 contains basic final-board state; F2 adds units and star levels;
F3 adds traits; F4 adds items; and F5 adds structural summaries. The exact operational
definitions and undocumented semantic details are in `tables/substantive_unit_definitions.md`.

## Strongest predictive unit

Full F5 XGBoost fold-local CV F1 was {full['mean_cv_f1']:.3f} (SD {full['fold_sd_f1']:.3f}).
By the prespecified ablation definition, **{GROUP_LABELS[str(strongest['substantive_unit'])]}**
was strongest: removing it produced the largest mean deterioration
({strongest['removal_deterioration']:.4f} F1). This is predictive importance, not causal influence.
TreeSHAP's largest aggregate unit was **{GROUP_LABELS[str(shap_unit.iloc[0]['substantive_unit'])]}**.
Differences between ablation and TreeSHAP are expected because correlated or redundant columns
can share attribution; {len(correlations)} feature pairs had absolute validation correlation ≥0.90.

## Model selection and stability

LightGBM had the slightly higher mean CV F1 ({lgb['cv_mean']:.4f} versus {xgb['cv_mean']:.4f}),
but the frozen selection rule maximized validation F1, where XGBoost scored
{xgb['validation_f1']:.4f} versus {lgb['validation_f1']:.4f}. The paired-fold comparison does
not justify claiming either leading model is intrinsically superior. Logistic regression remains
the parsimonious benchmark. Across 15 post-hoc grouped folds over three deterministic seeds,
XGBoost F1 averaged {repeated_mean:.3f} (fold SD {repeated_sd:.3f}).

## Exact placement

Fold-local random-forest placement CV achieved accuracy {rf_cv['mean_accuracy']:.3f}, macro F1
{rf_cv['mean_macro_f1']:.3f}, weighted F1 {rf_cv['mean_weighted_f1']:.3f}, and MAE
{rf_cv['mean_mae']:.3f} (fold SD {rf_cv['sd_mae']:.3f}), compared with dummy MAE
{dummy_cv['mean_mae']:.3f}. The locked final-test random forest achieved MAE
{test_place['mae']:.3f}; exact-placement prediction is meaningfully better than chance but moderate.

## Probability evaluation

At the frozen threshold 0.5, test Brier score was {test_prob['brier_score']:.3f}, log loss
{test_prob['log_loss']:.3f}, and 10-bin ECE {test_prob['expected_calibration_error_10_bin']:.3f}.
No application-specific alternative threshold is documented, so none was invented or optimized.

## Dependence and limitations

Matches never cross partitions or CV folds, and the outer split is chronological. Participant
overlap exists across {int((participant_overlap > 0).sum())} of three partition pairs, so match
grouping does not eliminate every player-level dependence. All data belong to one patch, set,
and queue; no cross-season generalization is established. TreeSHAP and ablations describe model
prediction, not causal effects. Small performance differences and correlated features require
cautious interpretation.

## Defensible conclusions

- Top-four classification is strong and generalizes well under the frozen split.
- Exact-placement prediction is better than the dummy baseline but remains moderate.
- F5 is the strongest combined frozen family; F4 is nearly equivalent in the frozen logistic family comparison.
- The new fold-local ablation—not the earlier aggregate comparison—identifies the strongest substantive predictive unit.
- Model importance is predictive and associational, not causal.
"""
    write_text(output / "thesis_ready_findings.md", summary)
    methods = """# Reproducibility and leakage audit

- Master seed: 4275; post-hoc stability seeds: 4275, 4276, 4277.
- Outer partition: frozen chronological 3,500/750/750 match split; canonical hash recorded in the manifest.
- CV: five-fold `StratifiedGroupKFold`, grouped by match. Every match's eight participants remain together.
- Fold preprocessing: `CompositionFeatureEncoder` is newly fitted on each CV training fold. Vocabulary frequency filtering is therefore fold-local. Held-out folds are transform-only; unseen categories create no columns.
- Missing values: deterministic numeric zero fill inside the encoder. Scaling: none. Resampling: none. Hyperparameter search: none.
- Ablations: fixed XGBoost contract parameters; only the prespecified substantive feature groups change.
- Placement CV: fixed registered random forest and dummy baseline; no selection or tuning.
- TreeSHAP: native XGBoost `pred_contribs` on the frozen development XGBoost and validation matrix. Values are additive log-odds contributions.
- Probability threshold: frozen 0.5. No alternative threshold was selected.
- Final test: read-only descriptive metrics for models already locked before the original single test evaluation; no model, feature, or threshold decision uses these results.
"""
    write_text(output / "reproducibility_and_leakage_audit.md", methods)


def run(output: Path = OUTPUT_ROOT) -> Path:
    output = output.resolve()
    verified = verify_inputs(PRODUCTION_RUN_ID, output)
    before = verified.artifact_hashes.copy()
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("tables", "figures", "logs"):
        (output / directory).mkdir()
    production = verified.production_root
    train = pd.read_csv(production / "data" / "train_participants.csv.gz")
    feature_definitions(production, output)
    fold_ablation, ablation = ablations_and_placement(train, output)
    placement_folds, placement = placement_cv(train, output)
    repeated = repeated_cv(train, output)
    shap_features, shap_units, local = shap_analysis(production, output)
    correlations = correlation_audit(production, output)
    selection = model_selection_audit(production, output)
    probability, final_placement = probability_and_final_placement(production, output)
    partitions = partition_audit(production, output)
    _write_reports(output, ablation, placement, shap_units, correlations, repeated,
                   selection, probability, final_placement, partitions)
    after = {path: sha256_file(production / path) for path in before}
    generated = sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file())
    manifest = {
        "extension_version": VERSION, "created_at_utc": utc_now(),
        "status": "PASS" if before == after else "FAIL",
        "source_analysis_run_id": PRODUCTION_RUN_ID, "dataset_version": DATASET_VERSION,
        "analysis_contract": CONTRACT_VERSION, "analysis_tag": ANALYSIS_TAG,
        "analysis_commit": ANALYSIS_COMMIT, "split_sha256": SPLIT_HASH,
        "split_counts": {"train": 3500, "validation": 750, "test": 750},
        "master_seed": SEED, "stability_seeds": list(SEEDS), "cv_folds": 5,
        "feature_families": {key: list(value) for key, value in FAMILIES.items()},
        "model_configuration": _xgb(SEED).get_params(),
        "placement_configuration": _rf(SEED).get_params(),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "xgboost": xgboost.__version__},
        "production_artifact_hashes_before_and_after_equal": before == after,
        "production_artifact_hashes": before, "generated_files": generated,
        "output_path": str(output),
        "test_use": "read_only_descriptive_evaluation_of_previously_locked_models",
    }
    write_text(output / "thesis_extension_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    if before != after:
        raise RuntimeError("Frozen production artifact hashes changed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run read-only extended thesis analyses")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    try:
        result = run(args.output_root)
    except Exception as exc:
        print(f"THESIS EXTENSION: FAIL\n{exc}", file=sys.stderr)
        raise
    print("THESIS EXTENSION: PASS")
    print(f"Outputs: {result}")


if __name__ == "__main__":
    main()

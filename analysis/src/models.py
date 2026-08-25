from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, mean_absolute_error, precision_score,
                             recall_score, roc_auc_score)

from .common import sha256_file, utc_now, write_csv, write_json
from .splits import cv_splits


def approved_model_registry() -> dict[str, Any]:
    """Machine-comparable scientific registry; seeds are governed separately."""
    return {
        "primary_registry": {
            "dummy": {"class": "sklearn.dummy.DummyClassifier", "parameters": {"strategy": "prior"}},
            "logistic_regression": {"class": "sklearn.linear_model.LogisticRegression", "parameters": {"max_iter": 2000}},
            "random_forest": {"class": "sklearn.ensemble.RandomForestClassifier", "parameters": {"n_estimators": 300, "min_samples_leaf": 2, "n_jobs": 1}},
            "xgboost": {"class": "xgboost.XGBClassifier", "parameters": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "logloss", "n_jobs": 1}},
            "lightgbm": {"class": "lightgbm.LGBMClassifier", "parameters": {"n_estimators": 300, "max_depth": -1, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": 1, "deterministic": True, "force_col_wise": True, "verbosity": -1}},
            "catboost": {"class": "catboost.CatBoostClassifier", "parameters": {"iterations": 300, "depth": 6, "learning_rate": 0.05, "loss_function": "Logloss", "verbose": False, "allow_writing_files": False, "thread_count": 1}},
        },
        "secondary_placement_registry": {
            "dummy": {"class": "sklearn.dummy.DummyClassifier", "parameters": {"strategy": "most_frequent"}},
            "random_forest": {"class": "sklearn.ensemble.RandomForestClassifier", "parameters": {"n_estimators": 300, "min_samples_leaf": 2, "n_jobs": 1}},
        },
    }


def approved_metric_registry() -> dict[str, Any]:
    return {
        "top4": {
            "primary_metric": "f1",
            "secondary_metrics": [
                "accuracy", "precision", "recall", "roc_auc", "pr_auc",
                "confusion_matrix",
            ],
        },
        "placement": {
            "primary_metric": "mae",
            "secondary_metrics": ["accuracy", "macro_f1", "weighted_f1"],
        },
    }


def run_development_selection(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_families: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    output_dir: Path,
    *, seed: int, folds: int,
) -> dict[str, Any]:
    """Fit on train and select once on validation; never accepts test data."""
    models_dir = output_dir / "models" / "development"
    tables_dir = output_dir / "tables"
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    train_x, validation_x = feature_families["F5"]
    estimators = _estimators(seed, require_boosters=True)
    validation_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    best_name = ""
    best_score = float("-inf")
    for name, estimator in estimators.items():
        assert estimator is not None
        fold_scores = []
        for fold, (fit_idx, held_idx) in enumerate(
            cv_splits(train.reset_index(drop=True), folds=folds, seed=seed), 1
        ):
            clone = _clone(estimator)
            clone.fit(train_x.iloc[fit_idx], train.iloc[fit_idx]["top4"].astype(int))
            predicted = clone.predict(train_x.iloc[held_idx])
            score = f1_score(train.iloc[held_idx]["top4"], predicted, zero_division=0)
            fold_scores.append(float(score))
            cv_rows.append({"target": "top4", "feature_family": "F5", "model": name,
                            "fold": fold, "f1": score})
        estimator.fit(train_x, train["top4"].astype(int))
        predicted = estimator.predict(validation_x)
        probability = _positive_probability(estimator, validation_x)
        metrics = _binary_metrics(validation["top4"].astype(int), predicted, probability)
        score = float(metrics["f1"])
        validation_rows.append({
            "target": "top4", "feature_family": "F5", "model": name,
            "partition": "validation", "cv_f1_mean": float(np.mean(fold_scores)), **metrics,
        })
        joblib.dump(estimator, models_dir / f"top4_f5_{name}.joblib")
        # Registry iteration order is the frozen deterministic tie-break.
        if score > best_score:
            best_name, best_score = name, score

    # Feature-family comparison is validation-only and cannot inspect test data.
    for family, (family_train, family_validation) in feature_families.items():
        model = LogisticRegression(max_iter=2000, random_state=seed)
        model.fit(family_train, train["top4"].astype(int))
        predicted = model.predict(family_validation)
        probability = _positive_probability(model, family_validation)
        validation_rows.append({
            "target": "top4", "feature_family": family,
            "model": "logistic_feature_family", "partition": "validation",
            **_binary_metrics(validation["top4"].astype(int), predicted, probability),
        })

    write_csv(tables_dir / "validation_metrics.csv", validation_rows)
    write_csv(tables_dir / "cross_validation_metrics.csv", cv_rows)
    selection = {
        "status": "locked",
        "locked_at_utc": utc_now(),
        "selected_task": "top4",
        "selected_feature_family": "F5",
        "selected_model": best_name,
        "selection_partition": "validation",
        "selection_metric": "f1",
        "selection_score": best_score,
        "classification_threshold": 0.5,
        "test_accessed": False,
        "registry_order": list(estimators),
    }
    write_json(output_dir / "selection_manifest.json", selection)
    return selection


def run_final_evaluation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    train_features: dict[str, pd.DataFrame],
    validation_features: dict[str, pd.DataFrame],
    test_features: dict[str, pd.DataFrame],
    output_dir: Path,
    *, selection: dict[str, Any], seed: int,
) -> dict[str, Any]:
    """Open test only after an immutable selection manifest has been written."""
    selection_path = output_dir / "selection_manifest.json"
    if selection.get("status") != "locked" or not selection_path.is_file():
        raise RuntimeError("Final test evaluation requires a locked selection manifest")
    if selection.get("test_accessed") is not False:
        raise RuntimeError("Test partition has already been accessed for this run")
    family = str(selection["selected_feature_family"])
    name = str(selection["selected_model"])
    estimator = _estimators(seed, require_boosters=True).get(name)
    if estimator is None:
        raise RuntimeError(f"Selected estimator is unavailable: {name}")
    development = pd.concat([train, validation], ignore_index=True)
    development_x = pd.concat([train_features[family], validation_features[family]],
                              ignore_index=True)
    estimator.fit(development_x, development["top4"].astype(int))
    predicted = estimator.predict(test_features[family])
    probability = _positive_probability(estimator, test_features[family])
    rows = [{
        "target": "top4", "feature_family": family, "model": name,
        "partition": "test", **_binary_metrics(test["top4"].astype(int), predicted, probability),
    }]
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, models_dir / "selected_top4_f5.joblib")

    # Secondary exact-placement baselines are fixed in advance, not test-selected.
    for secondary_name, secondary in _placement_estimators(seed).items():
        secondary.fit(development_x, development["placement"].astype(int))
        secondary_predicted = secondary.predict(test_features[family])
        rows.append({
            "target": "placement", "feature_family": family, "model": secondary_name,
            "partition": "test", **_placement_metrics(test["placement"], secondary_predicted),
        })
        joblib.dump(secondary, models_dir / f"placement_f5_{secondary_name}.joblib")
    metric_fields = [
        "target", "feature_family", "model", "partition", "accuracy", "precision",
        "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix", "mae",
        "macro_f1", "weighted_f1",
    ]
    write_csv(
        output_dir / "tables" / "final_test_metrics.csv",
        ({field: row.get(field, "") for field in metric_fields} for row in rows),
        metric_fields,
    )
    access = {
        **selection,
        "test_accessed": True,
        "test_accessed_at_utc": utc_now(),
        "test_evaluation_count": 1,
        "final_metrics_sha256": sha256_file(output_dir / "tables" / "final_test_metrics.csv"),
    }
    write_json(selection_path, access)
    return access


def _estimators(seed: int, require_boosters: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dummy": DummyClassifier(strategy="prior", random_state=seed),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                 n_jobs=1, random_state=seed),
    }
    missing: list[str] = []
    try:
        from xgboost import XGBClassifier
        result["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", n_jobs=1, random_state=seed,
        )
    except ImportError:
        missing.append("xgboost"); result["xgboost"] = None
    try:
        from lightgbm import LGBMClassifier
        result["lightgbm"] = LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1,
            deterministic=True, force_col_wise=True, verbosity=-1,
        )
    except ImportError:
        missing.append("lightgbm"); result["lightgbm"] = None
    try:
        from catboost import CatBoostClassifier
        result["catboost"] = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05, loss_function="Logloss",
            random_seed=seed, verbose=False, allow_writing_files=False, thread_count=1,
        )
    except ImportError:
        missing.append("catboost"); result["catboost"] = None
    if require_boosters and missing:
        raise RuntimeError(f"Production baseline dependencies are missing: {', '.join(missing)}")
    return result


def _placement_estimators(seed: int) -> dict[str, Any]:
    return {
        "dummy": DummyClassifier(strategy="most_frequent", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                 n_jobs=1, random_state=seed),
    }


def _clone(estimator: Any) -> Any:
    from sklearn.base import clone
    return clone(estimator)


def _positive_probability(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(x))[:, 1]
    return np.asarray(estimator.predict(x), dtype=float)


def _binary_metrics(y_true: Any, y_pred: Any, probability: Any) -> dict[str, Any]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "confusion_matrix": json.dumps(confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
                                       separators=(",", ":")),
    }


def _placement_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

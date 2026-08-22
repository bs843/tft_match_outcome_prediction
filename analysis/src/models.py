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

from .splits import cv_splits


def run_baselines(
    development: pd.DataFrame,
    test: pd.DataFrame,
    feature_families: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    output_dir: Path,
    *,
    seed: int,
    folds: int,
    require_boosters: bool,
) -> dict[str, Any]:
    models_dir = output_dir / "models"
    tables_dir = output_dir / "tables"
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    train_x, test_x = feature_families["F5"]
    estimators = _estimators(seed, require_boosters)
    result_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    for name, estimator in estimators.items():
        if estimator is None:
            result_rows.append({"target": "top4", "feature_family": "F5", "model": name,
                                "status": "dependency_unavailable"})
            continue
        fold_scores = []
        for fold, (fit_idx, validation_idx) in enumerate(cv_splits(development, folds=folds, seed=seed), 1):
            clone = _clone(estimator)
            clone.fit(train_x.iloc[fit_idx], development.iloc[fit_idx]["top4"].astype(int))
            predicted = clone.predict(train_x.iloc[validation_idx])
            score = f1_score(development.iloc[validation_idx]["top4"], predicted, zero_division=0)
            fold_scores.append(float(score))
            cv_rows.append({"target": "top4", "feature_family": "F5", "model": name,
                            "fold": fold, "f1": score})
        estimator.fit(train_x, development["top4"].astype(int))
        predicted = estimator.predict(test_x)
        probability = _positive_probability(estimator, test_x)
        metrics = _binary_metrics(test["top4"].astype(int), predicted, probability)
        result_rows.append({"target": "top4", "feature_family": "F5", "model": name,
                            "status": "complete", "cv_f1_mean": float(np.mean(fold_scores)), **metrics})
        joblib.dump(estimator, models_dir / f"top4_f5_{name}.joblib")

    # Exact placement is a secondary, eight-class baseline task.
    for name, estimator in _placement_estimators(seed).items():
        estimator.fit(train_x, development["placement"].astype(int))
        predicted = estimator.predict(test_x)
        result_rows.append({"target": "placement", "feature_family": "F5", "model": name,
                            "status": "complete", **_placement_metrics(test["placement"], predicted)})
        joblib.dump(estimator, models_dir / f"placement_f5_{name}.joblib")

    # Feature-family comparison uses one transparent model and the untouched test set.
    for family, (family_train, family_test) in feature_families.items():
        model = LogisticRegression(max_iter=2000, random_state=seed)
        model.fit(family_train, development["top4"].astype(int))
        predicted = model.predict(family_test)
        probability = _positive_probability(model, family_test)
        result_rows.append({"target": "top4", "feature_family": family,
                            "model": "logistic_feature_family", "status": "complete",
                            **_binary_metrics(test["top4"], predicted, probability)})

    pd.DataFrame(result_rows).to_csv(tables_dir / "baseline_metrics.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(tables_dir / "cross_validation_metrics.csv", index=False)
    return {"models": list(estimators), "results": result_rows}


def _estimators(seed: int, require_boosters: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dummy": DummyClassifier(strategy="prior", random_state=seed),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                 n_jobs=-1, random_state=seed),
    }
    missing: list[str] = []
    try:
        from xgboost import XGBClassifier
        result["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=seed,
        )
    except ImportError:
        missing.append("xgboost")
        result["xgboost"] = None
    try:
        from lightgbm import LGBMClassifier
        result["lightgbm"] = LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1,
            deterministic=True, force_col_wise=True, verbosity=-1,
        )
    except ImportError:
        missing.append("lightgbm")
        result["lightgbm"] = None
    try:
        from catboost import CatBoostClassifier
        result["catboost"] = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05, loss_function="Logloss",
            random_seed=seed, verbose=False, allow_writing_files=False,
            thread_count=-1,
        )
    except ImportError:
        missing.append("catboost")
        result["catboost"] = None
    if require_boosters and missing:
        raise RuntimeError(
            f"Production baseline dependencies are missing: {', '.join(missing)}; "
            "install requirements.txt"
        )
    return result


def _placement_estimators(seed: int) -> dict[str, Any]:
    return {
        "dummy": DummyClassifier(strategy="most_frequent", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                 n_jobs=-1, random_state=seed),
    }


def _clone(estimator: Any) -> Any:
    from sklearn.base import clone
    return clone(estimator)


def _positive_probability(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(x))[:, 1]
    return np.asarray(estimator.predict(x), dtype=float)


def _binary_metrics(y_true: Any, y_pred: Any, probability: Any) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "confusion_matrix": json.dumps(matrix, separators=(",", ":")),
    }


def _placement_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

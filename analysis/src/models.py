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

from .common import sha256_file, stable_hash, utc_now, write_csv, write_json
from .splits import cv_splits


TOP4_TASK = "top_four_classification"
PLACEMENT_TASK = "exact_placement"
SELECTION_MANIFESTS = {
    TOP4_TASK: "selection_manifest.json",
    PLACEMENT_TASK: "placement_selection_manifest.json",
}


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


def approved_selection_protocol() -> dict[str, Any]:
    """The task-aware v1.0.3 selection and test-isolation contract."""
    return {
        "exact_placement_validation_selection": True,
        "exact_placement_metric": "mae",
        "exact_placement_direction": "minimize",
        "exact_placement_tie_break": "secondary_placement_registry_order",
        "exact_placement_manifest": SELECTION_MANIFESTS[PLACEMENT_TASK],
        "task_aware_locked_manifests": True,
        "direct_multi_model_placement_test_evaluation": False,
    }


def run_development_selection(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_families: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    output_dir: Path,
    *, seed: int, folds: int, context: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Select both tasks on validation without accepting or reading test data."""
    models_dir = output_dir / "models" / "development"
    tables_dir = output_dir / "tables"
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    train_x, validation_x = feature_families["F5"]
    feature_hash = stable_hash({"family": "F5", "columns": list(train_x.columns)})
    registries = approved_model_registry()
    validation_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []

    top4_estimators = _estimators(seed, require_boosters=True)
    top4_best_name = ""
    top4_best_score = float("-inf")
    top4_metrics: dict[str, dict[str, Any]] = {}
    for name, estimator in top4_estimators.items():
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
        top4_metrics[name] = metrics
        score = float(metrics["f1"])
        validation_rows.append(_validation_row(
            context, task=TOP4_TASK, target="top4", model=name,
            metrics={"cv_f1_mean": float(np.mean(fold_scores)), **metrics},
            config=registries["primary_registry"][name], seed=seed,
            feature_hash=feature_hash,
        ))
        joblib.dump(estimator, models_dir / f"top4_f5_{name}.joblib")
        # Registry iteration order is the frozen deterministic tie-break.
        if score > top4_best_score:
            top4_best_name, top4_best_score = name, score

    # Feature-family comparison remains validation-only and ineligible for model selection.
    for family, (family_train, family_validation) in feature_families.items():
        model = LogisticRegression(max_iter=2000, random_state=seed)
        model.fit(family_train, train["top4"].astype(int))
        predicted = model.predict(family_validation)
        probability = _positive_probability(model, family_validation)
        validation_rows.append({
            **_validation_row(
                context, task=TOP4_TASK, target="top4", model="logistic_feature_family",
                metrics=_binary_metrics(validation["top4"].astype(int), predicted, probability),
                config={"family": family, "model": "logistic_regression"}, seed=seed,
                feature_hash=stable_hash({"family": family,
                                          "columns": list(family_train.columns)}),
            ),
            "feature_family": family,
            "selection_eligible": False,
        })

    placement_estimators = _placement_estimators(seed)
    placement_best_name = ""
    placement_best_score = float("inf")
    placement_metrics: dict[str, dict[str, Any]] = {}
    for name, estimator in placement_estimators.items():
        estimator.fit(train_x, train["placement"].astype(int))
        predicted = estimator.predict(validation_x)
        metrics = _placement_metrics(validation["placement"].astype(int), predicted)
        placement_metrics[name] = metrics
        score = float(metrics["mae"])
        validation_rows.append(_validation_row(
            context, task=PLACEMENT_TASK, target="placement", model=name,
            metrics=metrics, config=registries["secondary_placement_registry"][name],
            seed=seed, feature_hash=feature_hash,
        ))
        joblib.dump(estimator, models_dir / f"placement_f5_{name}.joblib")
        # Exact equality preserves registry order: dummy, then random forest.
        if score < placement_best_score:
            placement_best_name, placement_best_score = name, score

    for row in validation_rows:
        selected = (
            row["selection_eligible"]
            and ((row["task"] == TOP4_TASK and row["model"] == top4_best_name)
                 or (row["task"] == PLACEMENT_TASK and row["model"] == placement_best_name))
        )
        row["selected"] = selected

    validation_fields = list(dict.fromkeys(
        key for row in validation_rows for key in row
    ))
    write_csv(tables_dir / "validation_metrics.csv", validation_rows, validation_fields)
    write_csv(tables_dir / "cross_validation_metrics.csv", cv_rows)
    top4_selection = _selection_manifest(
        context, task=TOP4_TASK, target_definition="top4 = placement <= 4",
        selected_model=top4_best_name,
        model_contract=registries["primary_registry"][top4_best_name],
        validation_metrics=top4_metrics[top4_best_name], feature_hash=feature_hash,
        candidate_validation_results=top4_metrics,
        seed=seed, metric="f1", direction="maximize", tie_break="primary_registry_order",
        registry_order=list(top4_estimators), threshold=0.5,
    )
    placement_selection = _selection_manifest(
        context, task=PLACEMENT_TASK, target_definition="validated final placement 1-8",
        selected_model=placement_best_name,
        model_contract=registries["secondary_placement_registry"][placement_best_name],
        validation_metrics=placement_metrics[placement_best_name], feature_hash=feature_hash,
        candidate_validation_results=placement_metrics,
        seed=seed, metric="mae", direction="minimize",
        tie_break="secondary_placement_registry_order",
        registry_order=list(placement_estimators), threshold=None,
    )
    _write_new_locked_manifest(output_dir / SELECTION_MANIFESTS[TOP4_TASK], top4_selection)
    _write_new_locked_manifest(
        output_dir / SELECTION_MANIFESTS[PLACEMENT_TASK], placement_selection
    )
    return {TOP4_TASK: top4_selection, PLACEMENT_TASK: placement_selection}


def authorize_final_test_access(
    output_dir: Path,
    selections: dict[str, dict[str, Any]],
    run_manifest: dict[str, Any],
    run_manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    """Record both task access events before the caller opens the test payload."""
    verified = _verify_all_selections(output_dir, selections)
    existing = run_manifest.get("test_access_by_task", {})
    events: dict[str, dict[str, Any]] = {}
    for task in (TOP4_TASK, PLACEMENT_TASK):
        count = int(existing.get(task, {}).get("access_count", 0))
        if count != 0:
            raise RuntimeError(f"Final test for {task} has already been accessed")
        selection = verified[task]
        events[task] = {
            "task": task,
            "access_count": 1,
            "accessed_at_utc": utc_now(),
            "selected_model": selection["selected_model"],
            "selection_manifest": SELECTION_MANIFESTS[task],
            "selection_manifest_sha256": selection["manifest_sha256"],
        }
    run_manifest.update({
        "status": "final_evaluation_authorized",
        "test_accessed": True,
        "test_accessed_at_utc": min(event["accessed_at_utc"] for event in events.values()),
        "test_evaluation_count": 1,
        "test_access_by_task": events,
    })
    write_json(run_manifest_path, run_manifest)
    return events


def run_final_evaluation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    train_features: dict[str, pd.DataFrame],
    validation_features: dict[str, pd.DataFrame],
    test_features: dict[str, pd.DataFrame],
    output_dir: Path,
    *, selections: dict[str, dict[str, Any]], seed: int,
    access_events: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate only each task's locked winner after recorded authorization."""
    verified = _verify_all_selections(output_dir, selections)
    for task, selection in verified.items():
        event = access_events.get(task, {})
        if event.get("access_count") != 1:
            raise RuntimeError(f"Final test access was not recorded for {task}")
        if event.get("selected_model") != selection["selected_model"]:
            raise RuntimeError(f"Selected model differs from locked manifest for {task}")
        if event.get("selection_manifest_sha256") != selection["manifest_sha256"]:
            raise RuntimeError(f"Selection manifest identity differs for {task}")

    family = "F5"
    development = pd.concat([train, validation], ignore_index=True)
    development_x = pd.concat([train_features[family], validation_features[family]],
                              ignore_index=True)
    rows: list[dict[str, Any]] = []
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    top4_name = str(verified[TOP4_TASK]["selected_model"])
    top4_estimator = _estimators(seed, require_boosters=True).get(top4_name)
    if top4_estimator is None:
        raise RuntimeError(f"Selected top-four estimator is unavailable: {top4_name}")
    top4_estimator.fit(development_x, development["top4"].astype(int))
    top4_predicted = top4_estimator.predict(test_features[family])
    top4_probability = _positive_probability(top4_estimator, test_features[family])
    rows.append({
        "target": "top4", "feature_family": family, "model": top4_name,
        "partition": "test",
        **_binary_metrics(test["top4"].astype(int), top4_predicted, top4_probability),
    })
    joblib.dump(top4_estimator, models_dir / "selected_top4_f5.joblib")

    placement_name = str(verified[PLACEMENT_TASK]["selected_model"])
    placement_estimator = _placement_estimators(seed).get(placement_name)
    if placement_estimator is None:
        raise RuntimeError(f"Selected placement estimator is unavailable: {placement_name}")
    placement_estimator.fit(development_x, development["placement"].astype(int))
    placement_predicted = placement_estimator.predict(test_features[family])
    rows.append({
        "target": "placement", "feature_family": family, "model": placement_name,
        "partition": "test",
        **_placement_metrics(test["placement"].astype(int), placement_predicted),
    })
    joblib.dump(placement_estimator, models_dir / "selected_placement_f5.joblib")

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
    return {
        "test_accessed": True,
        "test_accessed_at_utc": min(
            event["accessed_at_utc"] for event in access_events.values()
        ),
        "test_evaluation_count": 1,
        "test_access_by_task": access_events,
        "final_metrics_sha256": sha256_file(output_dir / "tables" / "final_test_metrics.csv"),
    }


def verify_locked_selection(
    path: Path, *, expected_task: str, expected_selected_model: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Locked selection manifest is missing for {expected_task}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Locked selection manifest is invalid for {expected_task}") from exc
    if payload.get("status") != "locked":
        raise RuntimeError(f"Locked selection is required for {expected_task}")
    if payload.get("task") != expected_task:
        raise RuntimeError(
            f"Selection manifest task {payload.get('task')!r} cannot authorize {expected_task}"
        )
    supplied_hash = payload.get("manifest_sha256")
    hash_payload = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if supplied_hash != stable_hash(hash_payload):
        raise RuntimeError(f"Locked selection manifest integrity failed for {expected_task}")
    if payload.get("test_accessed") is not False:
        raise RuntimeError(f"Selection was not locked before test access for {expected_task}")
    if expected_selected_model is not None and payload.get("selected_model") != expected_selected_model:
        raise RuntimeError(f"Selected model differs from locked manifest for {expected_task}")
    return payload


def _verify_all_selections(
    output_dir: Path, selections: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if set(selections) != {TOP4_TASK, PLACEMENT_TASK}:
        raise RuntimeError("Locked selection is required independently for both production tasks")
    return {
        task: verify_locked_selection(
            output_dir / filename, expected_task=task,
            expected_selected_model=selections[task].get("selected_model"),
        )
        for task, filename in SELECTION_MANIFESTS.items()
    }


def _validation_row(
    context: dict[str, Any], *, task: str, target: str, model: str,
    metrics: dict[str, Any], config: dict[str, Any], seed: int, feature_hash: str,
) -> dict[str, Any]:
    return {
        "analysis_run_id": context["analysis_run_id"],
        "task": task,
        "target": target,
        "feature_family": "F5",
        "model": model,
        "partition": "validation",
        "config_identifier": stable_hash(config),
        "seed": seed,
        "feature_identity_sha256": feature_hash,
        "selection_eligible": True,
        "selected": False,
        **metrics,
    }


def _selection_manifest(
    context: dict[str, Any], *, task: str, target_definition: str,
    selected_model: str, model_contract: dict[str, Any],
    validation_metrics: dict[str, Any], feature_hash: str, seed: int,
    candidate_validation_results: dict[str, dict[str, Any]],
    metric: str, direction: str, tie_break: str, registry_order: list[str],
    threshold: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "locked",
        "task": task,
        "target_definition": target_definition,
        "selected_model": selected_model,
        "selected_hyperparameters": model_contract["parameters"],
        "selected_model_config_sha256": stable_hash(model_contract),
        "validation_metrics": validation_metrics,
        "candidate_validation_results": candidate_validation_results,
        "preprocessing_identity_sha256": context["preprocessing_sha256"],
        "feature_identity": {"family": "F5", "sha256": feature_hash},
        "seed": seed,
        "source_development_run_id": context["analysis_run_id"],
        "analysis_contract_version": context["contract_version"],
        "analysis_tag": context["analysis_tag"],
        "analysis_git_commit": context["git_commit"],
        "dataset_version": context["dataset_version"],
        "split_identity_sha256": context["split_hash"],
        "locked_at_utc": utc_now(),
        "test_accessed": False,
        "selection_rule": {
            "identity": f"{task}_{metric}_{direction}_{tie_break}",
            "version": "1",
            "partition": "validation",
            "primary_metric": metric,
            "direction": direction,
            "tie_break": tie_break,
            "registry_order": registry_order,
        },
    }
    if threshold is not None:
        payload["classification_threshold"] = threshold
    payload["manifest_sha256"] = stable_hash(payload)
    return payload


def _write_new_locked_manifest(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"Locked selection manifest already exists: {path}")
    write_json(path, payload)


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

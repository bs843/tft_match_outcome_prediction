from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import ensure_project_directories, load_config, project_path
from .provenance import config_hash, git_commit, new_run_id
from .utils import atomic_json_dump, utc_now_iso


def train_and_evaluate(
    config: dict[str, Any], *, experiment_id: str | None = None,
    data_run_id: str | None = None,
) -> Path:
    try:
        import joblib
        import numpy as np
        import pandas as pd
        import sklearn
        from sklearn.calibration import calibration_curve
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score,
        )
        from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Install the locked requirements before training") from exc

    processed = project_path(config, "processed") / data_run_id if data_run_id else project_path(config, "processed")
    frame = pd.read_csv(processed / "features.csv")
    required = {"match_hash", "participant_hash", "split", "top_four", "placement"}
    if missing := required.difference(frame.columns):
        raise RuntimeError(f"Feature matrix is missing: {', '.join(sorted(missing))}")
    target = frame["top_four"].astype(str).str.lower().eq("true").astype(int)
    feature_columns = [column for column in frame.columns if column not in required]
    prohibited = set(config["features"]["prohibited_columns"])
    if prohibited.intersection(feature_columns):
        raise RuntimeError("Prohibited leakage columns entered the feature matrix")
    X = frame[feature_columns].fillna(0).astype(float)
    train_mask, validation_mask, test_mask = [frame["split"].eq(name) for name in
                                               ("train", "validation", "test")]
    if not all(mask.any() for mask in (train_mask, validation_mask, test_mask)):
        raise RuntimeError("All three frozen splits must be nonempty")

    candidates: list[tuple[str, Any]] = [
        ("dummy_prior", DummyClassifier(strategy="prior")),
    ]
    for c_value in config["model"]["logistic_c_values"]:
        candidates.append((f"logistic_c_{c_value}", Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("model", LogisticRegression(C=float(c_value), max_iter=3000, random_state=int(config["sampling"]["random_seed"]))),
        ])))

    validation_results: dict[str, Any] = {}
    best_name, best_model, best_loss = "", None, float("inf")
    for name, model in candidates:
        model.fit(X.loc[train_mask], target.loc[train_mask])
        probabilities = model.predict_proba(X.loc[validation_mask])[:, 1]
        metrics = _metrics(target.loc[validation_mask].to_numpy(), probabilities)
        validation_results[name] = metrics
        if metrics["log_loss"] < best_loss:
            best_name, best_model, best_loss = name, model, metrics["log_loss"]
    assert best_model is not None

    cv_folds = min(int(config["model"]["cross_validation_folds"]),
                   frame.loc[train_mask, "match_hash"].nunique())
    cv_scores: list[float] = []
    if cv_folds >= 2 and best_name != "dummy_prior":
        splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True,
                                        random_state=int(config["sampling"]["random_seed"]))
        cv_scores = (-cross_val_score(
            best_model, X.loc[train_mask], target.loc[train_mask],
            groups=frame.loc[train_mask, "match_hash"], cv=splitter,
            scoring="neg_log_loss", n_jobs=1,
        )).tolist()

    development_mask = train_mask | validation_mask
    best_model.fit(X.loc[development_mask], target.loc[development_mask])
    test_probabilities = best_model.predict_proba(X.loc[test_mask])[:, 1]
    y_test = target.loc[test_mask].to_numpy()
    test_metrics = _metrics(y_test, test_probabilities)
    test_metrics["expected_calibration_error"] = _expected_calibration_error(y_test, test_probabilities)
    test_metrics.update(_within_match_metrics(
        frame.loc[test_mask, ["match_hash", "placement"]].copy(), test_probabilities
    ))
    test_groups = frame.loc[test_mask, "match_hash"].to_numpy()
    intervals = _match_bootstrap(
        y_test, test_probabilities, test_groups,
        iterations=int(config["model"]["bootstrap_iterations"]),
        seed=int(config["sampling"]["random_seed"]),
    )
    fraction_positive, mean_predicted = calibration_curve(y_test, test_probabilities, n_bins=10, strategy="quantile")

    experiment_id = experiment_id or new_run_id()
    run_dir = project_path(config, "models") / experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(best_model, run_dir / "fitted_pipeline.joblib")
    predictions = frame.loc[test_mask, ["match_hash", "participant_hash", "placement"]].copy()
    predictions["top_four"] = y_test
    predictions["probability_top_four"] = test_probabilities
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    report = {
        "experiment_id": experiment_id, "created_at": utc_now_iso(),
        "data_run_id": data_run_id,
        "study_type": "offline_post_match", "selected_model": best_name,
        "primary_metric": config["model"]["primary_metric"],
        "validation_models": validation_results, "training_cv_log_loss": cv_scores,
        "test_metrics": test_metrics, "test_match_bootstrap_95_ci": intervals,
        "calibration": {"mean_predicted_probability": mean_predicted.tolist(),
                        "fraction_positive": fraction_positive.tolist()},
        "feature_count": len(feature_columns), "feature_columns": feature_columns,
        "config_hash": config_hash(config), "git_commit": git_commit(),
        "python_packages": {"pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                            "joblib": joblib.__version__},
    }
    atomic_json_dump(report, run_dir / "metrics.json")
    return run_dir


def _metrics(y_true: Any, probabilities: Any) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score

    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions)),
    }


def _match_bootstrap(
    y_true: Any, probabilities: Any, groups: Any, *, iterations: int, seed: int
) -> dict[str, list[float]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    samples: dict[str, list[float]] = {key: [] for key in _metrics(y_true, probabilities)}
    for _ in range(iterations):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in selected])
        values = _metrics(np.asarray(y_true)[indices], np.asarray(probabilities)[indices])
        for key, value in values.items():
            samples[key].append(value)
    return {key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            for key, values in samples.items()}


def _expected_calibration_error(y_true: Any, probabilities: Any, bins: int = 10) -> float:
    import numpy as np

    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (p >= lower) & (p < upper if upper < 1 else p <= upper)
        if mask.any():
            value += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(value)


def _within_match_metrics(frame: Any, probabilities: Any) -> dict[str, float]:
    frame = frame.copy()
    frame["probability"] = probabilities
    hit_rates: list[float] = []
    correlations: list[float] = []
    for _, group in frame.groupby("match_hash"):
        selected = group.nlargest(4, "probability")
        hit_rates.append(float((selected["placement"] <= 4).mean()))
        correlation = group["probability"].corr(-group["placement"], method="spearman")
        if correlation == correlation:
            correlations.append(float(correlation))
    return {
        "within_match_top_four_hit_rate": sum(hit_rates) / len(hit_rates),
        "within_match_mean_spearman": sum(correlations) / len(correlations) if correlations else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the frozen offline model")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", required=True, help="Collection run providing the frozen feature matrix")
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    run_dir = train_and_evaluate(config, experiment_id=args.experiment_id, data_run_id=args.run_id)
    print(f"Saved experiment to {run_dir}")


if __name__ == "__main__":
    main()

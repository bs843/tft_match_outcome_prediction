from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, confusion_matrix, precision_recall_curve,
                             roc_curve)

from .common import observation_id, safe_binary_metrics, write_table
from .verify import VerifiedInputs


def build_final_diagnostics(verified: VerifiedInputs, output: Path, logger,
                            bootstrap_samples: int = 2000) -> dict[str, Any]:
    tables = output / "tables"
    figures = output / "figures"
    participants = pd.read_csv(verified.production_root / "data" / "test_participants.csv.gz")
    matrix = pd.read_csv(verified.production_root / "data" / "features" / "F5" /
                         "test_X.csv.gz")
    if len(participants) != 6000 or len(matrix) != len(participants):
        raise RuntimeError("Frozen test participant/features row alignment failed")
    model = joblib.load(verified.production_root / "modelling" / "models" /
                        "selected_top4_f5.joblib")
    probability = np.asarray(model.predict_proba(matrix))[:, 1]
    predicted = (probability >= 0.5).astype(int)
    truth = participants["top4"].astype(int).to_numpy()
    observed = safe_binary_metrics(truth, predicted, probability)
    stored = pd.read_csv(verified.production_root / "modelling" / "tables" /
                         "final_test_metrics.csv")
    stored = stored[stored["target"] == "top4"].iloc[0]
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        if abs(float(observed[metric]) - float(stored[metric])) > 1e-12:
            raise RuntimeError(f"Reproduced final-test {metric} differs from frozen metric")
    predictions = participants.copy()
    predictions["prediction"] = predicted
    predictions["probability"] = probability
    predictions["correct"] = predictions["prediction"].eq(predictions["top4"].astype(int))

    summary = confusion_summary(truth, predicted)
    tn, fp, fn, tp = (summary[key] for key in
                      ("true_negative", "false_positive", "false_negative", "true_positive"))
    confusion = pd.DataFrame(
        [[tn, fp], [fn, tp]], index=["true_bottom_four", "true_top_four"],
        columns=["predicted_bottom_four", "predicted_top_four"],
    )
    confusion.to_csv(tables / "final_test_confusion_matrix.csv")
    diagnostics = pd.DataFrame([summary])
    write_table(diagnostics, tables / "final_test_classification_diagnostics.csv",
                tables / "final_test_classification_diagnostics.md")
    _confusion_figure(confusion, figures / "final_test_confusion_matrix.png")

    fpr, tpr, roc_thresholds = roc_curve(truth, probability)
    roc_points = pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr,
                               "threshold": roc_thresholds})
    write_table(roc_points, tables / "final_test_roc_points.csv")
    _curve_figure(fpr, tpr, figures / "final_test_roc_curve.png", "False-positive rate",
                  "True-positive rate", "Final-test ROC curve", baseline="diagonal")
    precision, recall, pr_thresholds = precision_recall_curve(truth, probability)
    pr_points = pd.DataFrame({"recall": recall, "precision": precision,
                              "threshold": np.r_[pr_thresholds, np.nan]})
    write_table(pr_points, tables / "final_test_pr_points.csv")
    _curve_figure(recall, precision, figures / "final_test_precision_recall_curve.png",
                  "Recall", "Precision", "Final-test precision–recall curve",
                  baseline=float(truth.mean()))

    fraction_positive, mean_predicted = calibration_curve(
        truth, probability, n_bins=10, strategy="quantile"
    )
    bins = pd.qcut(probability, q=10, duplicates="drop")
    calibration = (pd.DataFrame({"bin": bins.astype(str), "truth": truth,
                                 "probability": probability})
                   .groupby("bin", observed=True, as_index=False)
                   .agg(n=("truth", "size"), observed_rate=("truth", "mean"),
                        mean_probability=("probability", "mean")))
    calibration["brier_score"] = brier_score_loss(truth, probability)
    write_table(calibration, tables / "final_test_calibration.csv")
    _calibration_figure(mean_predicted, fraction_positive,
                        figures / "final_test_calibration_curve.png")

    probability_summary = (predictions.assign(
        true_class=np.where(predictions["top4"].astype(int).eq(1), "top_four", "bottom_four")
    ).groupby("true_class")["probability"].describe(percentiles=[0.25, 0.5, 0.75])
      .reset_index().rename(columns={"50%": "median", "25%": "q1", "75%": "q3",
                                     "std": "sd"}))
    write_table(probability_summary, tables / "predicted_probability_summary.csv")
    _probability_figure(predictions, figures / "predicted_probability_distribution.png")

    errors_by_placement = (predictions.groupby("placement", as_index=False)
                           .agg(total_observations=("correct", "size"),
                                correctly_classified=("correct", "sum"),
                                mean_predicted_top_four_probability=("probability", "mean")))
    errors_by_placement["incorrectly_classified"] = (
        errors_by_placement["total_observations"]
        - errors_by_placement["correctly_classified"]
    )
    errors_by_placement["error_rate"] = (
        errors_by_placement["incorrectly_classified"] /
        errors_by_placement["total_observations"]
    )
    write_table(errors_by_placement, tables / "errors_by_final_placement.csv")
    _placement_error_figure(errors_by_placement, figures / "error_rate_by_placement.png")

    errors = predictions[~predictions["correct"]].copy()
    safe = safe_misclassification_table(errors)
    write_table(safe, tables / "final_test_misclassifications.csv")
    write_table(safe.sort_values("model_confidence", ascending=False),
                tables / "high_confidence_errors.csv")

    ci = match_cluster_bootstrap(predictions, samples=bootstrap_samples, seed=4275)
    write_table(ci, tables / "final_test_metric_ci.csv",
                tables / "final_test_metric_ci.md")
    logger.info("Reproduced frozen test predictions for %d rows; %d errors", len(predictions),
                len(errors))
    logger.info("Computed %d match-cluster bootstrap samples", bootstrap_samples)
    return {"predictions": predictions, "metrics": observed, "confusion": diagnostics,
            "calibration": calibration, "bootstrap": ci,
            "brier_score": float(brier_score_loss(truth, probability))}


def confusion_summary(y_true, y_pred) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn),
        "true_positive": int(tp), "specificity": float(tn / (tn + fp)),
        "sensitivity_recall": float(tp / (tp + fn)), "precision": float(tp / (tp + fp)),
        "negative_predictive_value": float(tn / (tn + fn)),
    }


def safe_misclassification_table(errors: pd.DataFrame) -> pd.DataFrame:
    frame = errors.copy()
    frame["observation_id"] = frame["participant_hash"].astype(str).map(observation_id)
    frame["model_confidence"] = np.maximum(frame["probability"], 1 - frame["probability"])
    safe_columns = [
        "observation_id", "top4", "prediction", "probability", "placement", "rank_tier",
        "rank_division", "rank_status", "platform_region", "regional_route",
        "model_confidence",
    ]
    return frame[safe_columns].rename(columns={"top4": "true_label",
                                               "placement": "final_placement"})


def match_cluster_bootstrap(predictions: pd.DataFrame, samples: int = 2000,
                            seed: int = 4275) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = {key: group.index.to_numpy() for key, group in predictions.groupby("match_hash")}
    match_ids = np.asarray(list(groups))
    rows: list[dict[str, float]] = []
    for _ in range(samples):
        sampled = rng.choice(match_ids, size=len(match_ids), replace=True)
        indices = np.concatenate([groups[match_id] for match_id in sampled])
        frame = predictions.loc[indices]
        metrics = safe_binary_metrics(frame["top4"].astype(int), frame["prediction"].astype(int),
                                      frame["probability"])
        rows.append({key: float(value) for key, value in metrics.items() if value is not None})
    bootstrap = pd.DataFrame(rows)
    point = safe_binary_metrics(predictions["top4"].astype(int),
                                predictions["prediction"].astype(int),
                                predictions["probability"])
    return pd.DataFrame([
        {"metric": metric, "point_estimate": float(point[metric]),
         "lower_95": float(bootstrap[metric].quantile(0.025)),
         "upper_95": float(bootstrap[metric].quantile(0.975)),
         "bootstrap_samples": samples, "resampling_unit": "match", "seed": seed}
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
    ])


def _confusion_figure(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.7, 5))
    image = ax.imshow(frame.to_numpy(), cmap="Blues")
    for (row, column), value in np.ndenumerate(frame.to_numpy()):
        ax.text(column, row, f"{value:,}", ha="center", va="center",
                color="white" if value > frame.to_numpy().max() / 2 else "black")
    ax.set_xticks([0, 1], ["Predicted bottom four", "Predicted top four"])
    ax.set_yticks([0, 1], ["True bottom four", "True top four"])
    ax.set_title("Final-test confusion matrix — locked XGBoost")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _curve_figure(x, y, path: Path, xlabel: str, ylabel: str, title: str,
                  baseline: str | float) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.plot(x, y, color="#245A9C", linewidth=2)
    if baseline == "diagonal":
        ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    else:
        ax.axhline(float(baseline), linestyle="--", color="grey",
                   label=f"Prevalence ({float(baseline):.3f})")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _calibration_figure(mean_predicted, fraction_positive, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.plot(mean_predicted, fraction_positive, "o-", color="#245A9C", label="Locked XGBoost")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed top-four rate",
           title="Final-test calibration (descriptive; no recalibration)", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _probability_figure(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bins = np.linspace(0, 1, 31)
    ax.hist(frame.loc[frame["top4"].eq(0), "probability"], bins=bins, alpha=0.6,
            density=True, label="True bottom four")
    ax.hist(frame.loc[frame["top4"].eq(1), "probability"], bins=bins, alpha=0.6,
            density=True, label="True top four")
    ax.axvline(0.5, linestyle="--", color="black", label="Frozen threshold")
    ax.set(xlabel="Predicted top-four probability", ylabel="Density",
           title="Final-test predicted-probability distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _placement_error_figure(frame: pd.DataFrame, path: Path) -> None:
    colors = ["#C44E52" if placement in (4, 5) else "#4C72B0"
              for placement in frame["placement"]]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.bar(frame["placement"].astype(str), frame["error_rate"], color=colors)
    ax.set(xlabel="Final placement", ylabel="Classification error rate",
           title="Final-test error rate by placement (4th/5th highlighted)", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)

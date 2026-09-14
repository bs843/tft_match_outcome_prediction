from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, mean_absolute_error,
                             precision_recall_fscore_support, precision_score, recall_score,
                             roc_auc_score)

from analysis.post_model.common import sha256_file, utc_now, write_table, write_text
from analysis.post_model.verify import verify_inputs
from analysis.thesis_extension.run import OUTPUT_ROOT, PRODUCTION_RUN_ID, SEED, _heatmap


ADDENDUM_FILES = (
    "tables/feature_family_validation_comparison.csv",
    "tables/unit_ablation_paired_deterioration.csv",
    "tables/top4_validation_test_comparison.csv",
    "tables/top4_match_bootstrap_ci.csv",
    "tables/placement_final_test_confusion_matrix.csv",
    "tables/placement_final_test_per_class.csv",
    "tables/placement_final_test_per_class.md",
    "tables/placement_match_bootstrap_ci.csv",
    "figures/placement_final_test_confusion_matrix.png",
    "uncertainty_and_final_evaluation_addendum.md",
    "thesis_extension_addendum_manifest.json",
)


def match_cluster_ci(frame: pd.DataFrame, metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
                     *, samples: int = 2000, seed: int = SEED) -> pd.DataFrame:
    matches = sorted(frame["match_hash"].astype(str).unique())
    grouped = {match: frame.loc[frame["match_hash"].astype(str) == match].reset_index(drop=True)
               for match in matches}
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in metrics}
    for _ in range(samples):
        selected = rng.choice(matches, size=len(matches), replace=True)
        sampled = pd.concat([grouped[match] for match in selected], ignore_index=True)
        truth = sampled["truth"].to_numpy()
        estimate = sampled["estimate"].to_numpy()
        for name, function in metrics.items():
            values[name].append(float(function(truth, estimate)))
    rows = []
    truth = frame["truth"].to_numpy(); estimate = frame["estimate"].to_numpy()
    for name, function in metrics.items():
        rows.append({"metric": name, "estimate": function(truth, estimate),
                     "ci95_low": np.quantile(values[name], .025),
                     "ci95_high": np.quantile(values[name], .975),
                     "bootstrap_samples": samples, "resampling_unit": "match", "seed": seed})
    return pd.DataFrame(rows)


def _binary_predictions(production: Path, partition: str) -> pd.DataFrame:
    model_path = (production / "modelling" / "models" /
                  ("development/top4_f5_xgboost.joblib" if partition == "validation"
                   else "selected_top4_f5.joblib"))
    model = joblib.load(model_path)
    x = pd.read_csv(production / "data" / "features" / "F5" / f"{partition}_X.csv.gz")
    source = pd.read_csv(production / "data" / f"{partition}_participants.csv.gz")
    probability = model.predict_proba(x)[:, 1]
    return pd.DataFrame({"match_hash": source["match_hash"].astype(str),
                         "truth": source["top4"].astype(int), "estimate": probability,
                         "prediction": (probability >= .5).astype(int)})


def finalize(output: Path = OUTPUT_ROOT) -> Path:
    output = output.resolve()
    if not (output / "thesis_extension_manifest.json").is_file():
        raise RuntimeError(f"Completed base extension is missing: {output}")
    existing = [str(output / name) for name in ADDENDUM_FILES if (output / name).exists()]
    if existing:
        raise RuntimeError("Addendum will not overwrite existing outputs: " + "; ".join(existing))
    verified = verify_inputs(PRODUCTION_RUN_ID, output, require_output_absent=False)
    before = verified.artifact_hashes.copy(); production = verified.production_root

    validation = pd.read_csv(production / "modelling" / "tables" / "validation_metrics.csv")
    families = validation.loc[(validation["model"] == "logistic_feature_family") &
                              (validation["task"] == "top_four_classification"),
                              ["feature_family", "accuracy", "precision", "recall", "f1",
                               "roc_auc", "pr_auc"]].sort_values("feature_family")
    families["change_in_f1_from_previous_family"] = families["f1"].diff()
    families["change_in_f1_from_f5"] = families["f1"] - float(
        families.loc[families["feature_family"] == "F5", "f1"].iloc[0])
    write_table(families, output / "tables" / "feature_family_validation_comparison.csv")

    folds = pd.read_csv(output / "tables" / "unit_ablation_fold_results.csv")
    full = folds.loc[folds["variant"] == "full_F5"].set_index("fold")["f1"]
    paired_rows = []
    for unit in ("basic", "units", "traits", "items", "structure"):
        removed = folds.loc[folds["variant"] == f"without_{unit}"].set_index("fold")["f1"]
        deterioration = full - removed
        interval = stats.t.interval(.95, len(deterioration) - 1,
                                    loc=deterioration.mean(), scale=stats.sem(deterioration))
        paired_rows.append({"substantive_unit": unit,
                            "mean_f1_deterioration_when_removed": deterioration.mean(),
                            "sd_paired_deterioration": deterioration.std(ddof=1),
                            "ci95_low": interval[0], "ci95_high": interval[1],
                            "folds_with_positive_deterioration": int((deterioration > 0).sum()),
                            "folds": len(deterioration),
                            "reproducible_positive_deterioration": bool(interval[0] > 0)})
    paired = pd.DataFrame(paired_rows).sort_values("mean_f1_deterioration_when_removed", ascending=False)
    write_table(paired, output / "tables" / "unit_ablation_paired_deterioration.csv")

    comparison_rows = []; binary_cis = []
    for partition in ("validation", "test"):
        pred = _binary_predictions(production, partition)
        truth = pred["truth"].to_numpy(); probability = pred["estimate"].to_numpy()
        label = pred["prediction"].to_numpy()
        comparison_rows.append({"partition": partition, "threshold": .5,
                                "accuracy": accuracy_score(truth, label),
                                "precision": precision_score(truth, label),
                                "recall": recall_score(truth, label), "f1": f1_score(truth, label),
                                "roc_auc": roc_auc_score(truth, probability),
                                "pr_auc": average_precision_score(truth, probability),
                                "brier_score": brier_score_loss(truth, probability),
                                "log_loss": log_loss(truth, probability, labels=[0, 1])})
        ci = match_cluster_ci(pred, {
            "f1": lambda y, p: f1_score(y, p >= .5),
            "accuracy": lambda y, p: accuracy_score(y, p >= .5),
            "roc_auc": lambda y, p: roc_auc_score(y, p),
            "brier_score": lambda y, p: brier_score_loss(y, p),
            "log_loss": lambda y, p: log_loss(y, p, labels=[0, 1]),
        })
        ci.insert(0, "partition", partition); binary_cis.append(ci)
    binary_comparison = pd.DataFrame(comparison_rows)
    binary_ci = pd.concat(binary_cis, ignore_index=True)
    write_table(binary_comparison, output / "tables" / "top4_validation_test_comparison.csv")
    write_table(binary_ci, output / "tables" / "top4_match_bootstrap_ci.csv")

    x = pd.read_csv(production / "data" / "features" / "F5" / "test_X.csv.gz")
    source = pd.read_csv(production / "data" / "test_participants.csv.gz")
    placement_model = joblib.load(production / "modelling" / "models" / "selected_placement_f5.joblib")
    placement_pred = np.asarray(placement_model.predict(x), dtype=int)
    labels = list(range(1, 9)); truth = source["placement"].astype(int).to_numpy()
    cm_values = confusion_matrix(truth, placement_pred, labels=labels)
    cm = pd.DataFrame(cm_values, index=[f"true_{x}" for x in labels],
                      columns=[f"pred_{x}" for x in labels])
    cm.to_csv(output / "tables" / "placement_final_test_confusion_matrix.csv")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, placement_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"placement": labels, "precision": precision, "recall": recall,
                              "f1": f1, "support": support})
    write_table(per_class, output / "tables" / "placement_final_test_per_class.csv",
                output / "tables" / "placement_final_test_per_class.md")
    _heatmap(cm_values, [str(x) for x in labels], [str(x) for x in labels],
             output / "figures" / "placement_final_test_confusion_matrix.png",
             "Final-test placement confusion matrix — locked random forest", "Predicted", "True")
    placement_frame = pd.DataFrame({"match_hash": source["match_hash"].astype(str),
                                    "truth": truth, "estimate": placement_pred})
    placement_ci = match_cluster_ci(placement_frame, {
        "mae": lambda y, p: mean_absolute_error(y, p),
        "accuracy": lambda y, p: accuracy_score(y, p),
        "within_1_accuracy": lambda y, p: np.mean(np.abs(y - p) <= 1),
        "within_2_accuracy": lambda y, p: np.mean(np.abs(y - p) <= 2),
    })
    write_table(placement_ci, output / "tables" / "placement_match_bootstrap_ci.csv")

    strongest = paired.iloc[0]
    f4 = families.loc[families["feature_family"] == "F4"].iloc[0]
    f5 = families.loc[families["feature_family"] == "F5"].iloc[0]
    text = f"""# Uncertainty and final-evaluation addendum

All intervals use 2,000 deterministic bootstrap resamples of complete matches. They are
descriptive uncertainty intervals; no result in this addendum changes model, feature, or
threshold selection.

- The strongest substantive predictive unit is `{strongest['substantive_unit']}`: its removal
  reduced F1 by {strongest['mean_f1_deterioration_when_removed']:.4f} on average
  (paired-fold 95% t interval {strongest['ci95_low']:.4f} to {strongest['ci95_high']:.4f});
  all {int(strongest['folds'])} folds deteriorated.
- Frozen logistic feature-family validation F1 was {f4['f1']:.4f} for F4 and {f5['f1']:.4f}
  for F5, a difference of {f5['f1'] - f4['f1']:.4f}; this supports “F4 nearly equivalent,”
  not a claim of statistically established superiority.
- `top4_match_bootstrap_ci.csv` reports validation and test uncertainty for discrimination,
  classification, and probability metrics.
- `placement_match_bootstrap_ci.csv` reports final-test uncertainty for the already locked
  placement model; the confusion matrix and per-placement table expose where moderate exact
  prediction remains difficult.
"""
    write_text(output / "uncertainty_and_final_evaluation_addendum.md", text)

    after = {path: sha256_file(production / path) for path in before}
    manifest = {"created_at_utc": utc_now(), "base_extension": "1.0.0",
                "status": "PASS" if before == after else "FAIL", "seed": SEED,
                "bootstrap_samples": 2000, "resampling_unit": "match",
                "production_artifact_hashes_before_and_after_equal": before == after,
                "files": {name: sha256_file(output / name) for name in ADDENDUM_FILES[:-1]}}
    write_text(output / "thesis_extension_addendum_manifest.json",
               json.dumps(manifest, indent=2, sort_keys=True))
    if before != after: raise RuntimeError("Frozen production artifacts changed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Add non-overwriting uncertainty outputs")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(); result = finalize(args.output_root)
    print("THESIS EXTENSION ADDENDUM: PASS"); print(f"Outputs: {result}")


if __name__ == "__main__": main()

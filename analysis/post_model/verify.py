from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.build_features import ID_FIELDS

from .common import (ANALYSIS_COMMIT, ANALYSIS_TAG, CONTRACT_VERSION, DATASET_VERSION,
                     PRODUCTION_RUN_ID, SEED, SPLIT_HASH, default_output_root,
                     production_root, read_json, sha256_file, stable_hash, utc_now,
                     write_json, write_text)


class PostModelVerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedInputs:
    production_root: Path
    manifest: dict[str, Any]
    contract: dict[str, Any]
    selection: dict[str, Any]
    placement_selection: dict[str, Any]
    artifact_hashes: dict[str, str]
    source_predictor_count: int
    encoded_feature_count: int
    verification_report: dict[str, Any]


def verify_inputs(run_id: str = PRODUCTION_RUN_ID, output_root: Path | None = None,
                  require_output_absent: bool = True) -> VerifiedInputs:
    root = production_root(run_id).resolve()
    output = (output_root or default_output_root()).resolve()
    failures: list[str] = []
    warnings: list[str] = []
    if not root.is_dir():
        raise PostModelVerificationError(f"Production run directory is missing: {root}")
    failures.extend(validate_output_path(output, root, require_output_absent))

    required = {
        "manifest": root / "run_manifest.json",
        "provenance": root / "provenance.json",
        "artifact_manifest": root / "artifact_manifest.csv",
        "split": root / "data" / "frozen_split_assignments.csv",
        "feature_vocabulary": root / "data" / "feature_vocabulary.json",
        "preprocessor": root / "data" / "fitted_train_encoder.joblib",
        "train_participants": root / "data" / "train_participants.csv.gz",
        "validation_participants": root / "data" / "validation_participants.csv.gz",
        "test_participants": root / "data" / "test_participants.csv.gz",
        "train_features": root / "data" / "features" / "F5" / "train_X.csv.gz",
        "validation_features": root / "data" / "features" / "F5" / "validation_X.csv.gz",
        "test_features": root / "data" / "features" / "F5" / "test_X.csv.gz",
        "cv_metrics": root / "modelling" / "tables" / "cross_validation_metrics.csv",
        "validation_metrics": root / "modelling" / "tables" / "validation_metrics.csv",
        "test_metrics": root / "modelling" / "tables" / "final_test_metrics.csv",
        "top4_selection": root / "modelling" / "selection_manifest.json",
        "placement_selection": root / "modelling" / "placement_selection_manifest.json",
        "top4_model": root / "modelling" / "models" / "selected_top4_f5.joblib",
        "placement_model": root / "modelling" / "models" / "selected_placement_f5.joblib",
        "development_top4_model": root / "modelling" / "models" / "development" /
                                  "top4_f5_xgboost.joblib",
    }
    for label, path in required.items():
        if not path.is_file():
            failures.append(f"Required {label} artifact is missing: {path}")

    if failures:
        raise PostModelVerificationError("; ".join(failures))
    manifest = read_json(required["manifest"])
    selection = read_json(required["top4_selection"])
    placement_selection = read_json(required["placement_selection"])
    contract_path = root.parents[1] / "config" / "analysis_contract.yaml"
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))

    expected_pairs = {
        "analysis run ID": (manifest.get("analysis_run_id"), run_id),
        "run status": (manifest.get("status"), "final_evaluation_complete"),
        "dataset version": (manifest.get("dataset_version"), DATASET_VERSION),
        "contract version": (manifest.get("analysis_contract_version"), CONTRACT_VERSION),
        "analysis tag": (manifest.get("analysis_tag"), ANALYSIS_TAG),
        "analysis commit": (manifest.get("analysis_git_commit"), ANALYSIS_COMMIT),
        "split hash": (manifest.get("split_hash"), SPLIT_HASH),
        "split counts": (manifest.get("split_counts"),
                         {"train": 3500, "validation": 750, "test": 750}),
        "master seed": (manifest.get("seed_policy", {}).get("master_seed"), SEED),
        "contract file version": (contract.get("analysis_contract_version"), CONTRACT_VERSION),
        "target field": (contract.get("targets", {}).get("top4", {}).get("field"), "top4"),
        "target derivation": (contract.get("targets", {}).get("top4", {}).get("derivation"),
                              "placement <= 4"),
        "selected top-four model": (selection.get("selected_model"), "xgboost"),
        "selection metric": (selection.get("selection_rule", {}).get("primary_metric"), "f1"),
        "selected placement model": (placement_selection.get("selected_model"),
                                     "random_forest"),
        "test evaluation count": (manifest.get("test_evaluation_count"), 1),
    }
    for label, (observed, expected) in expected_pairs.items():
        if observed != expected:
            failures.append(f"{label}: expected {expected!r}, observed {observed!r}")

    for task in ("top_four_classification", "exact_placement"):
        access = manifest.get("test_access_by_task", {}).get(task, {})
        if access.get("access_count") != 1:
            failures.append(f"Task {task} does not have exactly one recorded test access")
    for payload, task in ((selection, "top_four_classification"),
                          (placement_selection, "exact_placement")):
        supplied = payload.get("manifest_sha256")
        canonical = stable_hash({key: value for key, value in payload.items()
                                 if key != "manifest_sha256"})
        if supplied != canonical or payload.get("task") != task or payload.get("status") != "locked":
            failures.append(f"Locked selection integrity failed for {task}")

    artifact_hashes: dict[str, str] = {}
    with required["artifact_manifest"].open("r", encoding="utf-8-sig", newline="") as handle:
        artifact_rows = list(csv.DictReader(handle))
    for row in artifact_rows:
        path = root / row["relative_path"]
        if not path.is_file():
            failures.append(f"Manifested production artifact is missing: {path}")
            continue
        observed = sha256_file(path)
        artifact_hashes[row["relative_path"]] = observed
        if observed != row["sha256"]:
            failures.append(f"Production artifact hash mismatch: {row['relative_path']}")

    splits = pd.read_csv(required["split"], dtype=str)
    failures.extend(validate_split_frame(
        splits, {"test": 750, "train": 3500, "validation": 750}
    ))

    processed = (
        root.parents[1] / "data" / "processed" / manifest["collection_run_id"] /
        "features.csv"
    )
    with processed.open("r", encoding="utf-8-sig", newline="") as handle:
        fields = next(csv.reader(handle), [])
    source_predictors = [field for field in fields if field not in set(ID_FIELDS)]
    expected_predictors = int(contract["dataset"]["expected_predictors"])
    if len(source_predictors) != expected_predictors:
        failures.append(
            f"Source predictor count changed: {len(source_predictors)} != {expected_predictors}"
        )
    if any(field.startswith("rank_snapshot_") for field in source_predictors):
        failures.append("Prohibited rank snapshot field entered the source predictor matrix")
    encoded_columns = list(pd.read_csv(required["test_features"], nrows=0).columns)
    encoded_hash = stable_hash({"family": "F5", "columns": encoded_columns})
    if encoded_hash != selection.get("feature_identity", {}).get("sha256"):
        failures.append("Encoded F5 feature identity differs from locked selection")
    if any(column.startswith("rank_snapshot_") for column in encoded_columns):
        failures.append("Prohibited rank snapshot field entered encoded F5")

    try:
        tagged = subprocess.run(
            ["git", "rev-parse", f"{ANALYSIS_TAG}^{{commit}}"], cwd=root.parents[1],
            check=True, capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        if tagged != ANALYSIS_COMMIT:
            failures.append(f"Frozen analysis tag resolves to unexpected commit: {tagged}")
    except (OSError, subprocess.SubprocessError):
        warnings.append("Frozen analysis tag is unavailable locally; run provenance still records it")

    warnings.extend([
        "Fold-level CV stores F1 only; CV variability for other metrics is unavailable.",
        "Predicted probabilities were not persisted and will be reproduced from the frozen model and matrix.",
        "Source predictor count is 291; train-fitted F5 encoding contains 404 model columns.",
    ])
    report = {
        "status": "FAIL" if failures else "PASS",
        "verified_at_utc": utc_now(),
        "production_run_id": run_id,
        "dataset_version": manifest.get("dataset_version"),
        "analysis_contract": manifest.get("analysis_contract_version"),
        "analysis_tag": manifest.get("analysis_tag"),
        "analysis_commit": manifest.get("analysis_git_commit"),
        "split_hash": manifest.get("split_hash"),
        "split_counts": manifest.get("split_counts"),
        "predictor_count": len(source_predictors),
        "encoded_f5_feature_count": len(encoded_columns),
        "feature_manifest_hash": sha256_file(required["feature_vocabulary"]),
        "encoded_feature_identity_sha256": encoded_hash,
        "models_found": [str(required["top4_model"]), str(required["placement_model"])],
        "metrics_found": [str(required["cv_metrics"]), str(required["validation_metrics"]),
                          str(required["test_metrics"])],
        "preprocessing_found": str(required["preprocessor"]),
        "artifact_hashes_verified": len(artifact_hashes),
        "warnings": warnings,
        "failures": failures,
    }
    if failures:
        raise PostModelVerificationError("POST-MODEL ANALYSIS: NO-GO: " + "; ".join(failures))
    return VerifiedInputs(root, manifest, contract, selection, placement_selection,
                          artifact_hashes, len(source_predictors), len(encoded_columns), report)


def validate_output_path(output: Path, production: Path,
                         require_output_absent: bool = True) -> list[str]:
    failures = []
    if require_output_absent and output.exists():
        failures.append(f"Output directory already exists and will not be overwritten: {output}")
    try:
        output.resolve().relative_to(production.resolve())
        failures.append("Post-model output cannot be inside the frozen production run")
    except ValueError:
        pass
    return failures


def validate_split_frame(frame: pd.DataFrame,
                         expected_counts: dict[str, int]) -> list[str]:
    failures = []
    if not {"match_hash", "split"}.issubset(frame.columns):
        return ["Frozen split is missing match_hash or split"]
    counts = frame.groupby("split")["match_hash"].nunique().to_dict()
    if counts != expected_counts:
        failures.append(f"Frozen split counts differ: {counts}")
    if frame["match_hash"].duplicated().any() or frame["match_hash"].nunique() != sum(expected_counts.values()):
        failures.append("Frozen split has overlap, duplication, missing, or extra match IDs")
    return failures


def write_verification_reports(verified: VerifiedInputs, output_root: Path) -> None:
    write_json(output_root / "verification_report.json", verified.verification_report)
    report = verified.verification_report
    lines = [
        "# Post-model verification report", "", f"Status: **{report['status']}**", "",
        f"- Dataset: `{report['dataset_version']}`",
        f"- Contract: `{report['analysis_contract']}`",
        f"- Split hash: `{report['split_hash']}`",
        f"- Source predictors: {report['predictor_count']}",
        f"- Encoded F5 columns: {report['encoded_f5_feature_count']}",
        f"- Production hashes verified: {report['artifact_hashes_verified']}", "",
        "## Warnings", "",
        *[f"- {warning}" for warning in report["warnings"]], "",
        "## Failures", "",
        *([f"- {failure}" for failure in report["failures"]] or ["- None"]),
    ]
    write_text(output_root / "verification_report.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen production inputs for post-model analysis")
    parser.add_argument("--analysis-run", default=PRODUCTION_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    args = parser.parse_args()
    try:
        verified = verify_inputs(args.analysis_run, args.output_root)
        args.output_root.mkdir(parents=True, exist_ok=False)
        write_verification_reports(verified, args.output_root)
    except PostModelVerificationError as exc:
        print(f"POST-MODEL ANALYSIS: NO-GO\n{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print("POST-MODEL VERIFICATION: PASS")
    print(f"Report: {args.output_root / 'verification_report.json'}")


if __name__ == "__main__":
    main()

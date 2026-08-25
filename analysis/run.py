from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.config import PROJECT_ROOT, load_config

from .src.common import new_analysis_run_id, sha256_file, utc_now, write_json
from .src.contract import AnalysisContractError, run_analysis_preflight
from .src.dataset import build_participant_dataset
from .src.eda import run_eda
from .src.features import (build_feature_families, transform_feature_families,
                           validate_matrix_contract)
from .src.gate import AnalysisGateError
from .src.models import (PLACEMENT_TASK, TOP4_TASK, authorize_final_test_access,
                         run_development_selection, run_final_evaluation)
from .src.provenance import write_artifact_sidecar, write_provenance
from .src.roles import load_roles, validate_declared_columns
from .src.splits import FrozenSplitError, create_match_split
from .src.synthetic import synthetic_participant_dataset


ANALYSIS_ROOT = Path(__file__).resolve().parent
DETERMINISTIC_GZIP = {"method": "gzip", "compresslevel": 6, "mtime": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline TFT dissertation analysis pipeline")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--validation-run", help="Approved completed collection run ID")
    choice.add_argument("--dry-run", action="store_true", help="Synthetic engineering data only")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs"))
    args = parser.parse_args()
    try:
        output = run(validation_run=args.validation_run, dry_run=args.dry_run,
                     config_path=args.config, output_root=Path(args.output_root))
    except (AnalysisContractError, AnalysisGateError, FrozenSplitError) as exc:
        print(f"ANALYSIS BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"Analysis artifacts: {output}")


def run(*, validation_run: str | None, dry_run: bool, config_path: str | None,
        output_root: Path) -> Path:
    if dry_run:
        return _run_dry(output_root)
    config = load_config(config_path)
    run_id = new_analysis_run_id("analysis")
    output = output_root.resolve() / run_id
    try:
        preflight = run_analysis_preflight(config)
        if validation_run != preflight.approval.collection_run_id:
            raise AnalysisContractError(
                "Production CLI run ID differs from the contract-authorized collection"
            )
    except BaseException as exc:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "run_manifest.json", {
            "analysis_run_id": run_id, "created_at_utc": utc_now(),
            "status": "preflight_failed", "test_accessed": False,
            "error_class": type(exc).__name__,
        })
        raise

    contract = preflight.contract
    experiments_path = ANALYSIS_ROOT / "configs" / "experiments.yaml"
    columns_path = ANALYSIS_ROOT / "configs" / "columns.yaml"
    experiments = _yaml(experiments_path)
    column_rules = load_roles(columns_path)
    seed = int(contract["seeds"]["master_seed"])
    _set_seeds(seed)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "run_manifest.json"
    run_manifest: dict[str, Any] = {
        "analysis_run_id": run_id,
        "created_at_utc": utc_now(),
        "status": "development_complete" if False else "running",
        "dataset_version": preflight.approval.dataset_version,
        "collection_run_id": preflight.approval.collection_run_id,
        "collection_tag": preflight.approval.git_tag,
        "analysis_contract_version": contract["analysis_contract_version"],
        "analysis_tag": preflight.analysis_tag,
        "analysis_git_commit": preflight.analysis_commit,
        "git_clean": True,
        "configuration_hashes": {
            "analysis_contract": sha256_file(Path(contract["_path"])),
            "experiments": sha256_file(experiments_path),
            "columns": sha256_file(columns_path),
        },
        "split_hash": preflight.split_hash,
        "split_counts": contract["split"]["counts"],
        "seed_policy": contract["seeds"],
        "target_tasks": list(contract["targets"]),
        "test_accessed": False,
        "test_evaluation_count": 0,
        "test_access_by_task": {},
    }
    write_json(manifest_path, run_manifest)

    try:
        split_ids = {
            name: {row["match_hash"] for row in preflight.split_rows if row["split"] == name}
            for name in ("train", "validation", "test")
        }
        # Development reads only train and validation sanitized payloads.
        train = build_participant_dataset(preflight.approval, config, match_ids=split_ids["train"])
        validation = build_participant_dataset(
            preflight.approval, config, match_ids=split_ids["validation"]
        )
        validate_declared_columns(train.columns, column_rules)
        validate_declared_columns(validation.columns, column_rules)
        data_dir = output / "data"
        data_dir.mkdir()
        train.to_csv(data_dir / "train_participants.csv.gz", index=False,
                     compression=DETERMINISTIC_GZIP)
        validation.to_csv(data_dir / "validation_participants.csv.gz", index=False,
                          compression=DETERMINISTIC_GZIP)
        assignments = pd.DataFrame([
            {"match_hash": row["match_hash"], "split": row["split"]}
            for row in preflight.split_rows
        ]).sort_values("match_hash", kind="stable")
        split_path = data_dir / "frozen_split_assignments.csv"
        assignments.to_csv(split_path, index=False)

        encoder, families = build_feature_families(
            train, validation, experiments, column_rules,
        )
        for family, (train_x, validation_x) in families.items():
            validate_matrix_contract(list(train_x.columns), family=family, contract=contract)
            if list(train_x.columns) != list(validation_x.columns):
                raise RuntimeError(f"{family} validation schema differs from train")
            family_dir = data_dir / "features" / family
            family_dir.mkdir(parents=True)
            train_x.to_csv(family_dir / "train_X.csv.gz", index=False,
                           compression=DETERMINISTIC_GZIP)
            validation_x.to_csv(family_dir / "validation_X.csv.gz", index=False,
                                compression=DETERMINISTIC_GZIP)
        encoder_path = data_dir / "fitted_train_encoder.joblib"
        joblib.dump(encoder, encoder_path)
        write_json(data_dir / "feature_vocabulary.json", encoder.vocabulary_metadata())
        development = pd.concat([train, validation], ignore_index=True)
        eda = run_eda(development, output / "eda")
        selections = run_development_selection(
            train, validation, families, output / "modelling", seed=seed,
            folds=int(experiments["cross_validation_folds"]),
            context={
                "analysis_run_id": run_id,
                "contract_version": contract["analysis_contract_version"],
                "analysis_tag": preflight.analysis_tag,
                "git_commit": preflight.analysis_commit,
                "dataset_version": preflight.approval.dataset_version,
                "split_hash": preflight.split_hash,
                "preprocessing_sha256": sha256_file(encoder_path),
            },
        )
        top4_selection = selections[TOP4_TASK]
        placement_selection = selections[PLACEMENT_TASK]
        run_manifest.update({
            "status": "development_complete",
            "development_completed_at_utc": utc_now(),
            "task_selections": {
                TOP4_TASK: {
                    "selected_model": top4_selection["selected_model"],
                    "selection_metric": "f1",
                    "validation_score": top4_selection["validation_metrics"]["f1"],
                    "selection_manifest_sha256": top4_selection["manifest_sha256"],
                },
                PLACEMENT_TASK: {
                    "selected_model": placement_selection["selected_model"],
                    "selection_metric": "mae",
                    "validation_score": placement_selection["validation_metrics"]["mae"],
                    "selection_manifest_sha256": placement_selection["manifest_sha256"],
                },
            },
            "validation_results": {
                TOP4_TASK: top4_selection["candidate_validation_results"],
                PLACEMENT_TASK: placement_selection["candidate_validation_results"],
            },
        })
        write_json(manifest_path, run_manifest)

        # Verify both independently locked selections and record task-aware access
        # before opening any sanitized payload belonging to the test partition.
        access_events = authorize_final_test_access(
            output / "modelling", selections, run_manifest, manifest_path,
        )
        test = build_participant_dataset(preflight.approval, config, match_ids=split_ids["test"])
        validate_declared_columns(test.columns, column_rules)
        test.to_csv(data_dir / "test_participants.csv.gz", index=False,
                    compression=DETERMINISTIC_GZIP)
        test_features = transform_feature_families(encoder, test, experiments, column_rules)
        train_features = {name: pair[0] for name, pair in families.items()}
        validation_features = {name: pair[1] for name, pair in families.items()}
        for family, matrix in test_features.items():
            validate_matrix_contract(list(matrix.columns), family=family, contract=contract)
            if list(matrix.columns) != list(train_features[family].columns):
                raise RuntimeError(f"{family} test schema differs from frozen train schema")
            matrix.to_csv(data_dir / "features" / family / "test_X.csv.gz", index=False,
                          compression=DETERMINISTIC_GZIP)
        final = run_final_evaluation(
            train, validation, test, train_features, validation_features, test_features,
            output / "modelling", selections=selections, seed=seed,
            access_events=access_events,
        )
        run_manifest.update({
            "status": "final_evaluation_complete",
            "completed_at_utc": utc_now(),
            "test_accessed": final["test_accessed"],
            "test_accessed_at_utc": final["test_accessed_at_utc"],
            "test_evaluation_count": final["test_evaluation_count"],
            "test_access_by_task": final["test_access_by_task"],
            "final_test_metrics_sha256": final["final_metrics_sha256"],
            "required_artifacts": contract["production_artifacts"]["required"],
        })
        write_json(manifest_path, run_manifest)
        write_artifact_sidecar(split_path, artifact_type="frozen_match_group_split",
                               collection_run_id=preflight.approval.collection_run_id,
                               row_count=len(assignments), extra={"split_checksum": preflight.split_hash})
        write_json(output / "run_summary.json", {
            "analysis_mode": "PRODUCTION", "collection": preflight.approval.collection_run_id,
            "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
            "feature_families": list(families), "eda": eda,
            "selected_models": {
                TOP4_TASK: top4_selection["selected_model"],
                PLACEMENT_TASK: placement_selection["selected_model"],
            },
        })
        write_provenance(
            output, analysis_run_id=run_id, mode="PRODUCTION",
            collection={
                "run_id": preflight.approval.collection_run_id,
                "dataset_version": preflight.approval.dataset_version,
                "schema_version": preflight.approval.schema_version,
                "git_tag": preflight.approval.git_tag,
                "analysis_tag": preflight.analysis_tag,
                "analysis_contract_version": contract["analysis_contract_version"],
                "input_checksums": preflight.approval.checksums,
            },
            config_checksums=run_manifest["configuration_hashes"],
            split_checksum=preflight.split_hash,
        )
        artifact_paths = _verify_required_artifacts(
            output, contract["production_artifacts"]["required"]
        )
        run_manifest["output_artifact_paths"] = artifact_paths
        write_json(manifest_path, run_manifest)
        # Refresh provenance/artifact checksums after the final manifest update.
        write_provenance(
            output, analysis_run_id=run_id, mode="PRODUCTION",
            collection={
                "run_id": preflight.approval.collection_run_id,
                "dataset_version": preflight.approval.dataset_version,
                "schema_version": preflight.approval.schema_version,
                "git_tag": preflight.approval.git_tag,
                "analysis_tag": preflight.analysis_tag,
                "analysis_contract_version": contract["analysis_contract_version"],
                "input_checksums": preflight.approval.checksums,
            },
            config_checksums=run_manifest["configuration_hashes"],
            split_checksum=preflight.split_hash,
        )
        return output
    except BaseException as exc:
        run_manifest.update({"status": "failed", "failed_at_utc": utc_now(),
                             "error_class": type(exc).__name__})
        write_json(manifest_path, run_manifest)
        raise


def _run_dry(output_root: Path) -> Path:
    """Synthetic, explicitly non-production path; legacy splitting is isolated here."""
    output = output_root.resolve() / new_analysis_run_id("dry-run")
    output.mkdir(parents=True, exist_ok=False)
    frame = synthetic_participant_dataset(20)
    legacy = create_match_split(frame, seed=4275, test_fraction=0.2)
    write_json(output / "NON_ANALYTICAL_TEST_ONLY.json", {
        "status": "NON-ANALYTICAL TEST/DEVELOPMENT ONLY",
        "warning": "Legacy engineering split; unreachable from production analysis.",
        "legacy_80_20": True,
        "production_valid": False,
    })
    write_json(output / "run_manifest.json", {
        "status": "development_complete", "production_valid": False,
        "test_accessed": False, "matches": frame["match_hash"].nunique(),
        "engineering_development_matches": legacy.development["match_hash"].nunique(),
    })
    return output


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _verify_required_artifacts(output: Path, required: list[str]) -> list[str]:
    missing = [relative for relative in required if not (output / relative).is_file()]
    if missing:
        raise RuntimeError(f"Production run is missing contracted artifacts: {missing}")
    return sorted(required)


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Configuration must be a mapping: {path}")
    return value


if __name__ == "__main__":
    main()

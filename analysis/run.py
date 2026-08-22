from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from src.config import PROJECT_ROOT, load_config

from .src.common import new_analysis_run_id, sha256_file, write_json
from .src.dataset import build_participant_dataset
from .src.eda import run_eda
from .src.features import build_feature_families
from .src.gate import AnalysisGateError, require_analysis_approval
from .src.models import run_baselines
from .src.provenance import (AnalysisCodeStateError, require_versioned_clean_analysis_code,
                             write_artifact_sidecar, write_provenance)
from .src.roles import load_roles, validate_declared_columns
from .src.splits import create_match_split
from .src.synthetic import synthetic_participant_dataset


ANALYSIS_ROOT = Path(__file__).resolve().parent
DETERMINISTIC_GZIP = {"method": "gzip", "compresslevel": 6, "mtime": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline TFT dissertation analysis pipeline")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--validation-run", help="Approved completed collection run ID")
    choice.add_argument("--dry-run", action="store_true", help="Use synthetic engineering data only")
    parser.add_argument("--config", default=None, help="Study configuration path")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs"))
    args = parser.parse_args()
    try:
        output = run(
            validation_run=args.validation_run,
            dry_run=args.dry_run,
            config_path=args.config,
            output_root=Path(args.output_root),
        )
    except (AnalysisGateError, AnalysisCodeStateError) as exc:
        print(f"ANALYSIS BLOCKED: {exc}" if isinstance(exc, AnalysisCodeStateError) else str(exc),
              file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"Analysis artifacts: {output}")


def run(*, validation_run: str | None, dry_run: bool, config_path: str | None,
        output_root: Path) -> Path:
    config = load_config(config_path)
    experiments_path = ANALYSIS_ROOT / "configs" / "experiments.yaml"
    columns_path = ANALYSIS_ROOT / "configs" / "columns.yaml"
    experiments = _yaml(experiments_path)
    column_rules = load_roles(columns_path)
    analysis_run_id = new_analysis_run_id("dry-run" if dry_run else "analysis")
    output = output_root.resolve() / analysis_run_id

    if dry_run:
        output.mkdir(parents=True, exist_ok=False)
        mode = "NON-ANALYTICAL TEST/DEVELOPMENT ONLY"
        write_json(output / "run_state.json", {"status": "running", "analysis_mode": mode})
        frame = synthetic_participant_dataset()
        collection = {"run_id": "SYNTHETIC", "approved": False,
                      "notice": mode}
        write_json(output / "NON_ANALYTICAL_TEST_ONLY.json", {
            "status": mode,
            "warning": "These synthetic outputs are not thesis results and must not be interpreted.",
        })
    else:
        # No output directory is created until the authoritative gate approves
        # the real collection and returns sanitized paths.
        approval = require_analysis_approval(config, str(validation_run))
        analysis_commit = require_versioned_clean_analysis_code(PROJECT_ROOT)
        output.mkdir(parents=True, exist_ok=False)
        mode = "PRODUCTION"
        write_json(output / "run_state.json", {
            "status": "running", "analysis_mode": mode,
            "collection_run_id": approval.collection_run_id,
        })
        frame = build_participant_dataset(approval, config)
        collection = {
            "run_id": approval.collection_run_id,
            "approved": True,
            "accepted_matches": approval.accepted_count,
            "participants": approval.participant_count,
            "dataset_version": approval.dataset_version,
            "schema_version": approval.schema_version,
            "git_commit": approval.git_commit,
            "git_tag": approval.git_tag,
            "analysis_git_commit": analysis_commit,
            "input_checksums": approval.checksums,
        }

    validate_declared_columns(frame.columns, column_rules)
    data_dir = output / "data"
    data_dir.mkdir()
    participant_path = data_dir / "participant_final_boards.csv.gz"
    frame.to_csv(participant_path, index=False, compression=DETERMINISTIC_GZIP)
    split = create_match_split(frame, seed=int(experiments["random_seed"]),
                               test_fraction=float(experiments["test_fraction"]))
    split_path = data_dir / "match_split_assignments.csv"
    split.assignments.to_csv(split_path, index=False)
    regional_groups = (frame[["match_hash", "platform_region"]].drop_duplicates()
                       .merge(split.assignments, on="match_hash", how="left", validate="one_to_one")
                       .sort_values("match_hash"))
    regional_groups.to_csv(data_dir / "regional_subgroup_manifest.csv", index=False)
    split.assignments.loc[split.assignments["split"] == "development", ["match_hash"]].to_csv(
        data_dir / "development_match_ids.csv", index=False
    )
    split.assignments.loc[split.assignments["split"] == "test", ["match_hash"]].to_csv(
        data_dir / "test_match_ids.csv", index=False
    )
    split.development.to_csv(data_dir / "development_participants.csv.gz", index=False,
                             compression=DETERMINISTIC_GZIP)
    split.test.to_csv(data_dir / "test_participants.csv.gz", index=False,
                      compression=DETERMINISTIC_GZIP)
    encoder, families = build_feature_families(
        split.development, split.test, experiments, column_rules,
    )
    for family, (train_x, test_x) in families.items():
        family_dir = data_dir / "features" / family
        family_dir.mkdir(parents=True)
        train_x.to_csv(family_dir / "development_X.csv.gz", index=False,
                       compression=DETERMINISTIC_GZIP)
        test_x.to_csv(family_dir / "test_X.csv.gz", index=False,
                      compression=DETERMINISTIC_GZIP)
    write_json(data_dir / "feature_vocabulary.json", encoder.vocabulary_metadata())
    write_json(data_dir / "split_metadata.json", {
        "random_seed": experiments["random_seed"],
        "test_fraction": experiments["test_fraction"],
        "development_matches": int(split.development["match_hash"].nunique()),
        "test_matches": int(split.test["match_hash"].nunique()),
        "split_checksum": split.checksum,
        "grouping_key": "match_hash",
    })
    eda = run_eda(frame, output / "eda")
    models = run_baselines(
        split.development, split.test, families, output / "modelling",
        seed=int(experiments["random_seed"]),
        folds=int(experiments["cross_validation_folds"]),
        require_boosters=not dry_run,
    )
    source_run_id = str(collection["run_id"])
    write_artifact_sidecar(participant_path, artifact_type="participant_final_board_dataset",
                           collection_run_id=source_run_id, row_count=len(frame))
    write_artifact_sidecar(split_path, artifact_type="frozen_match_group_split",
                           collection_run_id=source_run_id, row_count=len(split.assignments),
                           extra={"split_checksum": split.checksum})
    metrics_path = output / "modelling" / "tables" / "baseline_metrics.csv"
    write_artifact_sidecar(metrics_path, artifact_type="baseline_evaluation",
                           collection_run_id=source_run_id)
    write_json(output / "run_summary.json", {
        "analysis_mode": mode,
        "collection": collection,
        "rows": len(frame),
        "matches": int(frame["match_hash"].nunique()),
        "feature_families": list(families),
        "eda": eda,
        "baseline_models": models["models"],
    })
    write_json(output / "run_state.json", {
        "status": "complete", "analysis_mode": mode,
        "collection_run_id": collection["run_id"],
    })
    write_provenance(
        output, analysis_run_id=analysis_run_id, mode=mode, collection=collection,
        config_checksums={
            "study_config_sha256": sha256_file(Path(config["_config_path"])),
            "experiments_config_sha256": sha256_file(experiments_path),
            "column_roles_sha256": sha256_file(columns_path),
        },
        split_checksum=split.checksum,
    )
    return output


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Configuration must be a mapping: {path}")
    return value


if __name__ == "__main__":
    main()

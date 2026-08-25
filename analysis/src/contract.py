from __future__ import annotations

import subprocess
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.config import PROJECT_ROOT
from src.build_features import ID_FIELDS

from .common import read_csv, sha256_file, sha256_text_file
from .gate import AnalysisApproval, require_analysis_approval
from .models import (approved_metric_registry, approved_model_registry,
                     approved_selection_protocol)
from .splits import canonical_split_hash, validate_frozen_assignments


class AnalysisContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnalysisPreflight:
    contract: dict[str, Any]
    approval: AnalysisApproval
    split_manifest: Path
    split_rows: tuple[dict[str, str], ...]
    split_hash: str
    analysis_commit: str
    analysis_tag: str
    checks: tuple[str, ...]


def load_contract(path: Path | None = None) -> dict[str, Any]:
    contract_path = path or PROJECT_ROOT / "config" / "analysis_contract.yaml"
    try:
        value = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AnalysisContractError(f"Analysis contract cannot be loaded: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisContractError("Analysis contract must be a mapping")
    if value.get("analysis_contract_version") != "1.0.3":
        raise AnalysisContractError(
            "Analysis contract version must be 1.0.3; a new release is required"
        )
    value["_path"] = str(contract_path.resolve())
    return value


def run_analysis_preflight(
    config: dict[str, Any], *, contract_path: Path | None = None,
    verify_release: bool = True,
) -> AnalysisPreflight:
    contract = load_contract(contract_path)
    checks: list[str] = ["contract_version"]
    dataset = _mapping(contract, "dataset")
    _equal(config["project"]["dataset_version"], dataset["required_dataset_version"],
           "dataset version")
    _equal(config["project"]["schema_version"], dataset["required_schema_version"],
           "schema version")
    run_id = str(dataset["collection_run_id"])
    approval = require_analysis_approval(config, run_id)
    _equal(approval.dataset_version, dataset["required_dataset_version"], "approved dataset version")
    _equal(approval.schema_version, dataset["required_schema_version"], "approved schema version")
    _equal(approval.git_tag, dataset["required_collection_tag"], "collection provenance tag")
    _equal(approval.accepted_count, int(dataset["expected_matches"]), "accepted match count")
    _equal(approval.participant_count, int(dataset["expected_participants"]), "participant count")
    _equal(approval.checksums["accepted_manifest_content_sha256"],
           dataset["accepted_manifest_content_sha256"], "accepted manifest identity")
    checks.extend(["dataset_version", "collection_provenance", "accepted_cohort"])

    processed_features = (PROJECT_ROOT / config["paths"]["processed"] / run_id / "features.csv")
    if not processed_features.is_file():
        raise AnalysisContractError("Frozen processed feature table is missing")
    with processed_features.open("r", encoding="utf-8-sig", newline="") as handle:
        fields = next(csv.reader(handle), [])
    predictors = [field for field in fields if field not in set(ID_FIELDS)]
    _equal(len(predictors), int(dataset["expected_predictors"]), "frozen predictor count")
    if any(field.startswith("rank_snapshot_") for field in predictors):
        raise AnalysisContractError("Rank snapshot variables entered the frozen predictor matrix")
    checks.append("frozen_feature_schema")

    _verify_bound_file(dataset, "collection_manifest", "collection_manifest_sha256")
    _verify_bound_file(dataset, "dataset_manifest", "dataset_manifest_sha256")
    configuration = _mapping(contract, "configuration")
    _verify_bound_text_file(configuration, "experiments_file", "experiments_sha256")
    _verify_bound_text_file(configuration, "columns_file", "columns_sha256")
    checks.extend(["dataset_manifest_hash", "experiments_hash", "columns_hash"])

    experiments = _yaml(PROJECT_ROOT / configuration["experiments_file"])
    _verify_scientific_configuration(contract, experiments)
    checks.extend(["feature_contract", "model_registry", "metric_registry", "seed_policy",
                   "test_access_policy", "exact_placement_validation_selection",
                   "task_aware_locked_manifests", "placement_test_isolation"])

    split = _mapping(contract, "split")
    split_path = PROJECT_ROOT / split["manifest"]
    _verify_hash(split_path, str(split["manifest_file_sha256"]), "frozen split manifest")
    rows = read_csv(split_path)
    expected_ids = {row["match_hash"] for row in read_csv(approval.validated_manifest)}
    expected_counts = {name: int(value) for name, value in split["counts"].items()}
    observed_hash = validate_frozen_assignments(
        rows, expected_match_ids=expected_ids, expected_counts=expected_counts,
        dataset_version=str(dataset["required_dataset_version"]),
    )
    _equal(observed_hash, split["canonical_assignment_sha256"], "canonical split hash")
    _equal(expected_counts, {"train": 3500, "validation": 750, "test": 750},
           "authoritative split counts")
    _equal(split.get("legacy_80_20_production_allowed"), False, "legacy split prohibition")
    checks.extend(["split_manifest_hash", "split_counts", "split_overlap", "cohort_reconciliation"])

    commit, tag = ("candidate-unreleased", str(contract["release"]["required_analysis_tag"]))
    if verify_release:
        commit, tag = verify_analysis_release(contract)
        checks.extend(["git_cleanliness", "analysis_commit", "analysis_tag"])
    return AnalysisPreflight(
        contract=contract, approval=approval, split_manifest=split_path,
        split_rows=tuple(rows), split_hash=observed_hash,
        analysis_commit=commit, analysis_tag=tag, checks=tuple(checks),
    )


def verify_analysis_release(contract: dict[str, Any]) -> tuple[str, str]:
    release = _mapping(contract, "release")
    tag = str(release["required_analysis_tag"])
    if release.get("required_analysis_commit") != "resolve_from_required_analysis_tag":
        raise AnalysisContractError("Unsupported analysis commit resolution policy")
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise AnalysisContractError("Production analysis requires a clean Git working tree")
    try:
        tag_type = _git("cat-file", "-t", f"refs/tags/{tag}")
        tagged_commit = _git("rev-parse", f"{tag}^{{commit}}")
    except AnalysisContractError as exc:
        raise AnalysisContractError(f"Required analysis release tag is missing: {tag}") from exc
    if release.get("require_annotated_tag") is True and tag_type != "tag":
        raise AnalysisContractError(f"Required analysis tag is not annotated: {tag}")
    head = _git("rev-parse", "HEAD")
    if head != tagged_commit:
        raise AnalysisContractError(
            f"Analysis HEAD mismatch: expected {tagged_commit}, observed {head}"
        )
    return head, tag


def _verify_scientific_configuration(contract: dict[str, Any], experiments: dict[str, Any]) -> None:
    split = contract["split"]
    _equal(experiments.get("split_source"), "frozen_collection_manifest", "split source")
    _equal(experiments.get("expected_split_counts"), split["counts"], "experiment split counts")
    _equal(int(experiments.get("random_seed", -1)), int(contract["seeds"]["master_seed"]),
           "master seed")
    _equal(experiments.get("primary_target"), "top4", "primary target")
    _equal(experiments.get("secondary_target"), "placement", "secondary target")
    _equal(experiments.get("rank_features_in_primary_models"), False, "rank exclusion")
    _equal(experiments.get("primary_model_selection_metric"),
           contract["selection"]["primary_task_metric"], "selection metric")
    observed_families = {
        name: details.get("groups") for name, details in experiments.get("feature_families", {}).items()
    }
    _equal(observed_families, contract["features"]["families"], "feature-family registry")
    expected_models = list(contract["models"]["primary_registry"])
    _equal(expected_models,
           ["dummy", "logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"],
           "primary model registry")
    registry = approved_model_registry()
    _equal(contract["models"]["primary_registry"], registry["primary_registry"],
           "primary model definitions")
    _equal(contract["models"]["secondary_placement_registry"],
           registry["secondary_placement_registry"], "secondary model definitions")
    metrics = approved_metric_registry()
    for target, expected in metrics.items():
        _equal(contract["targets"][target]["primary_metric"], expected["primary_metric"],
               f"{target} primary metric")
        _equal(contract["targets"][target]["secondary_metrics"], expected["secondary_metrics"],
               f"{target} secondary metrics")
    _equal(contract["test_access"]["policy"],
            "final_evaluation_only_after_selection_manifest_written", "test access policy")
    protocol = approved_selection_protocol()
    exact = contract["selection"].get("exact_placement", {})
    _equal(exact.get("enabled"), protocol["exact_placement_validation_selection"],
           "exact-placement validation selection")
    _equal(exact.get("primary_metric"), protocol["exact_placement_metric"],
           "exact-placement selection metric")
    _equal(exact.get("direction"), protocol["exact_placement_direction"],
           "exact-placement metric direction")
    _equal(exact.get("tie_break"), protocol["exact_placement_tie_break"],
           "exact-placement tie break")
    _equal(exact.get("registry_order"), list(registry["secondary_placement_registry"]),
           "exact-placement registry order")
    _equal(exact.get("locked_manifest"), protocol["exact_placement_manifest"],
           "exact-placement locked manifest")
    test_access = contract["test_access"]
    _equal(test_access.get("task_aware_locked_manifests"),
           protocol["task_aware_locked_manifests"], "task-aware locked manifests")
    _equal(test_access.get("exact_placement_requires_locked_selection"), True,
           "exact-placement locked selection gate")
    _equal(test_access.get("exact_placement_manifest_integrity_required"), True,
           "exact-placement manifest integrity gate")
    _equal(test_access.get("direct_multi_model_placement_test_evaluation"),
           protocol["direct_multi_model_placement_test_evaluation"],
           "direct multi-model placement test prohibition")
    required = contract.get("production_artifacts", {}).get("required", [])
    for artifact in (
        "modelling/placement_selection_manifest.json",
        "modelling/models/selected_placement_f5.joblib",
    ):
        if artifact not in required:
            raise AnalysisContractError(f"Required placement artifact is missing: {artifact}")


def _verify_bound_file(section: dict[str, Any], path_key: str, hash_key: str) -> None:
    _verify_hash(PROJECT_ROOT / str(section[path_key]), str(section[hash_key]), path_key)


def _verify_bound_text_file(section: dict[str, Any], path_key: str, hash_key: str) -> None:
    _verify_text_hash(PROJECT_ROOT / str(section[path_key]), str(section[hash_key]), path_key)


def _verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise AnalysisContractError(f"Contract-bound file is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise AnalysisContractError(
            f"{label} hash mismatch for {path}: expected {expected}, observed {observed}; "
            "a new analysis-contract version/tag is required"
        )


def _verify_text_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise AnalysisContractError(f"Contract-bound file is missing: {path}")
    observed = sha256_text_file(path)
    if observed != expected:
        raise AnalysisContractError(
            f"{label} canonical-text hash mismatch for {path}: expected {expected}, "
            f"observed {observed}; a new analysis-contract version/tag is required"
        )


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AnalysisContractError(f"Configuration is not a mapping: {path}")
    return value


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise AnalysisContractError(f"Analysis contract section is missing: {key}")
    return result


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AnalysisContractError(f"{label}: expected {expected!r}, observed {actual!r}")


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True,
            text=True, timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise AnalysisContractError(f"Git release check failed: {args}") from exc

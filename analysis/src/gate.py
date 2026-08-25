from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.provenance import accepted_manifest_checksum, config_hash
from src.utils import read_csv_rows

from .common import sha256_file


class AnalysisGateError(RuntimeError):
    """Raised when a collection is not authorised for thesis analysis."""


@dataclass(frozen=True)
class AnalysisApproval:
    collection_run_id: str
    collection_manifest: Path
    validation_summary: Path
    validated_manifest: Path
    validated_files: tuple[Path, ...]
    validated_file_map: dict[str, Path]
    accepted_count: int
    participant_count: int
    dataset_version: str
    schema_version: str
    git_commit: str
    git_tag: str
    checksums: dict[str, str]


def require_analysis_approval(config: dict[str, Any], run_id: str) -> AnalysisApproval:
    """Fail closed unless *run_id* is the complete, validated analytical cohort.

    This function deliberately reads collection metadata but returns only paths to
    sanitized validation outputs. Downstream analysis must use ``validated_files``.
    """
    root = PROJECT_ROOT
    target = _positive_int(config.get("sampling", {}).get("target_accepted_matches"),
                           "sampling.target_accepted_matches")
    manifest_path = root / config["paths"]["manifests"] / f"collection_run_{run_id}.json"
    manifest = _required_json(manifest_path, "collection manifest")
    _equal(manifest.get("run_id"), run_id, "collection manifest run_id")
    _equal(manifest.get("status"), "complete", "collection manifest status")
    _equal(manifest.get("requested_accepted_match_target"), target,
           "requested accepted-match target")
    _equal(manifest.get("dataset_version"), config["project"]["dataset_version"],
           "dataset version")
    _equal(manifest.get("schema_version"), config["project"]["schema_version"],
           "schema version")
    _equal(manifest.get("study_patch"), "17.8", "study patch")
    expected_platforms = [
        {"platform_region": row["platform_region"], "regional_route": row["regional_route"]}
        for row in config.get("platforms", []) if row.get("enabled")
    ]
    _equal(manifest.get("enabled_platforms"), expected_platforms, "enabled platform provenance")
    _equal(manifest.get("config_hash"), config_hash(config), "configuration checksum")
    if not manifest.get("accepted_match_manifest_id"):
        _fail("accepted_match_manifest_id is absent")
    if not manifest.get("accepted_match_manifest_sha256"):
        _fail("accepted_match_manifest_sha256 is absent")
    accepted_stage = _positive_int(manifest.get("stage_counts", {}).get("accepted_matches"),
                                   "stage_counts.accepted_matches")
    if accepted_stage < target:
        _fail(f"only {accepted_stage} accepted matches; {target} required")

    accepted_path = root / config["paths"]["accepted_match_ids"]
    accepted_rows = [r for r in read_csv_rows(
        accepted_path, ("run_id", "match_id", "platform_region", "regional_route",
                        "raw_sha256", "validation_accepted"))
                     if r.get("run_id") == run_id]
    match_ids = [r["match_id"] for r in accepted_rows]
    if len(match_ids) != target or len(set(match_ids)) != target:
        _fail(f"accepted manifest must contain exactly {target} unique matches")
    if any(not row["platform_region"] or not row["regional_route"] or
           len(row["raw_sha256"]) != 64 or
           row["validation_accepted"].strip().lower() != "true" for row in accepted_rows):
        _fail("accepted manifest contains incomplete region/checksum/validation provenance")
    checksum = accepted_manifest_checksum(run_id, accepted_rows)
    _equal(manifest.get("accepted_match_manifest_sha256"), checksum,
           "accepted manifest checksum")

    report_path = root / config["paths"]["validation_reports"] / run_id / "validation_summary.json"
    report = _required_json(report_path, "validation summary")
    _equal(report.get("collection_status") in {"PASS", "PASS_WITH_WARNINGS"}, True,
           "validation status approval")
    for field in ("critical_failures", "invalid_matches", "invalid_participants",
                  "duplicate_match_count", "missing_required_field_count",
                  "end_result_blank_count", "end_result_mismatch_count", "schema_errors"):
        _equal(report.get(field), 0, f"validation {field}")
    counts = report.get("counts")
    if not isinstance(counts, dict):
        _fail("validation counts are absent")
    for field in ("raw_checksum_failures", "non_17_8_matches", "unknown_patch_matches"):
        _equal(counts.get(field), 0, f"validation counts.{field}")
    _equal(counts.get("accepted_manifest_present"), True,
           "validation accepted_manifest_present")
    _equal(report.get("patch_17_8_percentage"), 100.0, "Patch 17.8 percentage")
    _equal(report.get("valid_matches"), target, "validated match count")
    expected_participants = target * int(config["scope"]["expected_participants"])
    _equal(report.get("valid_participants"), expected_participants,
           "validated participant count")
    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        _fail("validation metadata are absent")
    _equal(metadata.get("collection_id"), run_id, "validation collection_id")
    _equal(metadata.get("collection_manifest_status"), "complete",
           "validation collection manifest status")
    _equal(metadata.get("expected_patch"), "17.8", "validation expected patch")
    _equal(metadata.get("schema_version"), config["project"]["schema_version"],
           "validation schema version")
    _equal(metadata.get("enabled_platforms"), expected_platforms,
           "validation enabled platform provenance")
    if config.get("validation", {}).get("rank_snapshot_required"):
        _equal(report.get("rank_snapshot_match_rate"), 100.0,
               "required rank snapshot match rate")

    validated_root = (root / config["paths"]["validated"] / run_id).resolve()
    validated_manifest = validated_root / "validated_manifest.csv"
    validated_rows = read_csv_rows(
        validated_manifest,
        ("run_id", "match_hash", "source_sha256", "validated_file", "validation_status"),
    )
    if len(validated_rows) != target:
        _fail(f"validated manifest has {len(validated_rows)} rows; {target} required")
    hashes = [row["match_hash"] for row in validated_rows]
    if len(set(hashes)) != target:
        _fail("validated manifest contains duplicate match hashes")
    files: list[Path] = []
    file_map: dict[str, Path] = {}
    for row in validated_rows:
        _equal(row.get("run_id"), run_id, "validated manifest run_id")
        _equal(row.get("validation_status"), "valid", "validated row status")
        candidate = (validated_root / row["validated_file"]).resolve()
        if validated_root not in candidate.parents:
            _fail("validated manifest contains a path outside its run directory")
        if not candidate.is_file():
            _fail(f"validated file is missing: {candidate.name}")
        files.append(candidate)
        file_map[row["match_hash"]] = candidate

    return AnalysisApproval(
        collection_run_id=run_id,
        collection_manifest=manifest_path,
        validation_summary=report_path,
        validated_manifest=validated_manifest,
        validated_files=tuple(files),
        validated_file_map=file_map,
        accepted_count=target,
        participant_count=expected_participants,
        dataset_version=str(manifest["dataset_version"]),
        schema_version=str(manifest["schema_version"]),
        git_commit=str(manifest.get("git_commit") or "unavailable"),
        git_tag=str(manifest.get("git_tag") or "unavailable"),
        checksums={
            "collection_manifest_sha256": sha256_file(manifest_path),
            "validation_summary_sha256": sha256_file(report_path),
            "validated_manifest_sha256": sha256_file(validated_manifest),
            "accepted_manifest_content_sha256": checksum,
        },
    )


def _required_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisGateError(f"ANALYSIS BLOCKED: invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} is not an object")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        _fail(f"{label} is missing or invalid")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        _fail(f"{label} is missing or invalid")
    if parsed <= 0:
        _fail(f"{label} is missing or invalid")
    return parsed


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        _fail(f"{label}: expected {expected!r}, observed {actual!r}")


def _fail(reason: str) -> None:
    raise AnalysisGateError(f"ANALYSIS BLOCKED: {reason}")

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import DEFAULT_CONFIG_PATH, load_config, project_path
from ..provenance import accepted_manifest_checksum
from ..utils import get_hash_key, hash_identifier, read_csv_rows, safe_filename
from . import VALIDATION_CODE_VERSION
from .validate_matches import validate_match
from .validate_rank_snapshots import validate_rank_snapshots
from .validate_schema import validate_rank_feature_exclusion, validate_schema
from .validation_report import Issue, ValidationSummary, final_status, write_csv, write_json, write_markdown

LOGGER = logging.getLogger("tft.validation")
ISSUE_FIELDS = ["severity", "scope", "reason_code", "reason", "match_hash", "participant_hash", "field"]


def run_validation(
    config: dict[str, Any], *, run_id: str,
    report_dir: Path | None = None, validated_dir: Path | None = None,
) -> tuple[ValidationSummary, list[Issue]]:
    raw_dir = project_path(config, "raw_matches")
    report_dir = report_dir or project_path(config, "validation_reports") / run_id
    validated_dir = validated_dir or project_path(config, "validated") / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    rules = _rules(config)
    key = get_hash_key(config)
    pseudonym = lambda value: hash_identifier(value, key)
    issues: list[Issue] = []

    manifest_path = project_path(config, "manifests") / f"collection_run_{run_id}.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Collection manifest not found for run {run_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    collection_complete = manifest.get("status") == "complete"
    if not collection_complete:
        issues.append(Issue("critical", "collection", "collection_not_complete",
                            f"Collection manifest status is {manifest.get('status', 'unknown')}"))

    status_path = project_path(config, "collection_status")
    accepted_path = project_path(config, "accepted_match_ids")
    rank_path = project_path(config, "rank_snapshots")
    for name, path in {
        "collection_status": status_path, "accepted_match_ids": accepted_path,
        "raw_matches": raw_dir, "rank_snapshots": rank_path,
    }.items():
        if not path.exists():
            severity = "warning" if name == "rank_snapshots" and not rules["rank_snapshot_required"] else "critical"
            issues.append(Issue(severity, "collection", "required_input_missing",
                                f"Required input {name} is missing: {path}"))

    status_rows = [row for row in read_csv_rows(status_path) if row.get("run_id") == run_id]
    manifest_accepted = [row for row in read_csv_rows(accepted_path) if row.get("run_id") == run_id]
    accepted_manifest_sha256 = accepted_manifest_checksum(run_id, manifest_accepted)
    recorded_accepted_sha256 = str(manifest.get("accepted_match_manifest_sha256") or "")
    if recorded_accepted_sha256 and recorded_accepted_sha256 != accepted_manifest_sha256:
        issues.append(Issue(
            "critical", "collection", "accepted_manifest_checksum_mismatch",
            "Frozen accepted-match partition differs from its collection manifest checksum",
        ))
    if manifest_accepted:
        cohort_ids = [row["match_id"] for row in manifest_accepted]
    else:
        cohort_ids = [row["match_id"] for row in status_rows
                      if row.get("validation_accepted", "").lower() == "true"]
        issues.append(Issue("critical", "collection", "accepted_manifest_missing_for_run",
                            "No frozen accepted-match partition exists; using provisional status decisions for audit only"))
    if not cohort_ids:
        issues.append(Issue("critical", "collection", "empty_analytical_cohort",
                            "No accepted or provisionally accepted matches were found"))

    duplicate_status_ids = [match_id for match_id, count in Counter(
        row.get("match_id", "") for row in status_rows
    ).items() if match_id and count > 1]
    for match_id in duplicate_status_ids:
        issues.append(Issue("critical", "collection", "duplicate_collection_status_match",
                            "Collection status contains duplicate match rows", match_hash=pseudonym(match_id)))

    LOGGER.info("Auditing %d downloaded rows and %d analytical matches", len(status_rows), len(cohort_ids))
    checksum_failures = 0
    for row in status_rows:
        match_id = row.get("match_id", "")
        if not row.get("raw_file"):
            continue
        path = raw_dir / f"{safe_filename(match_id)}.json"
        if not path.exists():
            issues.append(Issue("critical", "collection", "raw_file_missing",
                                "Status row references a missing raw response", match_hash=pseudonym(match_id)))
            continue
        expected_sha = row.get("raw_sha256", "")
        actual_sha = _sha256(path)
        if expected_sha and expected_sha != actual_sha:
            checksum_failures += 1
            issues.append(Issue("critical", "collection", "raw_checksum_mismatch",
                                "Raw response checksum differs from collection status", match_hash=pseudonym(match_id)))

    status_by_id = {row.get("match_id", ""): row for row in status_rows}
    match_results: list[dict[str, Any]] = []
    seen_match_ids: set[str] = set()
    payloads: list[dict[str, Any]] = []
    for match_id in cohort_ids:
        path = raw_dir / f"{safe_filename(match_id)}.json"
        if not path.exists():
            issues.append(Issue("critical", "match", "raw_file_missing",
                                "Analytical match raw file is missing", match_hash=pseudonym(match_id)))
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(Issue("critical", "match", "raw_json_unparseable",
                                f"Raw response cannot be parsed: {type(exc).__name__}", match_hash=pseudonym(match_id)))
            continue
        payloads.append(payload)
        result = validate_match(payload, expected_file_match_id=match_id,
                                hash_identifier=pseudonym, rules=rules,
                                seen_match_ids=seen_match_ids)
        result["source_sha256"] = status_by_id.get(match_id, {}).get("raw_sha256") or _sha256(path)
        match_results.append(result)
        issues.extend(result["issues"])

    schema_report, schema_issues = validate_schema(payloads)
    issues.extend(schema_issues)
    feature_columns = _feature_columns(config, run_id)
    issues.extend(validate_rank_feature_exclusion(config, feature_columns))

    participant_hashes: set[str] = set()
    latest_match: dict[str, int] = {}
    total_participants = 0
    invalid_participants = 0
    missingness: Counter[str] = Counter()
    for result in match_results:
        missingness.update(result["missingness"])
        for participant in result["participants"]:
            total_participants += 1
            participant_hash = participant["participant_hash"]
            if participant_hash:
                participant_hashes.add(participant_hash)
                timestamp = result.get("game_datetime_ms") or 0
                latest_match[participant_hash] = max(latest_match.get(participant_hash, 0), timestamp)
            if not participant["valid"]:
                invalid_participants += 1

    issues.extend(_validate_interim_join(config, run_id, match_results))
    rank_rows = [row for row in read_csv_rows(rank_path) if row.get("run_id") == run_id] if rank_path.exists() else []
    rank_result = validate_rank_snapshots(
        rank_rows, expected_participants=participant_hashes,
        participant_latest_match_ms=latest_match, required=rules["rank_snapshot_required"],
    )
    issues.extend(rank_result["issues"])

    missingness_rows = [{
        "field": field, "missing_count": missingness.get(field, 0),
        "missing_percentage": round(100 * missingness.get(field, 0) / total_participants, 4)
        if total_participants else 0.0,
        "classification": "required" if field in {"puuid", "placement", "level", "units", "traits"} else "optional",
    } for field in sorted(missingness)]

    patch_counts = Counter(result.get("patch", {}).get("classification", "unknown_patch") for result in match_results)
    valid_matches = sum(result.get("valid", False) for result in match_results)
    invalid_matches = len(match_results) - valid_matches
    critical = [issue for issue in issues if issue.severity == "critical"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    status = final_status(issues)
    rank_summary = rank_result["summary_rows"][0]
    summary = ValidationSummary(
        collection_status=status,
        total_matches=len(match_results), valid_matches=valid_matches,
        invalid_matches=invalid_matches, total_participants=total_participants,
        valid_participants=total_participants - invalid_participants,
        invalid_participants=invalid_participants,
        patch_17_8_percentage=round(100 * patch_counts["patch_17_8"] / len(match_results), 4)
        if match_results else 0.0,
        duplicate_match_count=sum(issue.reason_code in {"duplicate_match_id", "duplicate_collection_status_match"} for issue in issues),
        invalid_placement_count=sum(issue.reason_code in {"invalid_placement", "invalid_placement_structure"} for issue in issues),
        missing_required_field_count=sum(issue.reason_code == "missing_required_field" for issue in issues),
        end_result_blank_count=sum(len(result["participants"]) for result in match_results
                                   if result.get("end_result_blank") or result.get("end_result_null")),
        end_result_mismatch_count=sum(not result.get("end_result_valid", False) for result in match_results),
        rank_snapshot_match_rate=rank_summary["match_rate_percentage"],
        schema_errors=sum(issue.scope == "schema" and issue.severity == "critical" for issue in issues),
        warnings=len(warnings), critical_failures=len(critical),
        metadata={
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "collection_id": run_id,
            "collection_manifest_status": manifest.get("status", "unknown"),
            "collection_pipeline_version": manifest.get("dataset_version", "unknown"),
            "schema_version": manifest.get("schema_version", config["project"]["schema_version"]),
            "configuration_version": manifest.get(
                "configuration_version", config.get("provenance", {}).get("configuration_version", "unknown")
            ),
            "configuration_hash": manifest.get("config_hash", "unknown"),
            "git_commit_sha": manifest.get("git_commit", "unavailable"),
            "git_tag": manifest.get("git_tag", "unavailable"),
            "accepted_match_manifest_id": manifest.get(
                "accepted_match_manifest_id", f"accepted_match_ids.csv#run_id={run_id}"
            ),
            "accepted_match_manifest_sha256": accepted_manifest_sha256,
            "validation_code_version": VALIDATION_CODE_VERSION,
            "expected_patch": rules["expected_patch"],
            "collection_start_timestamp": config["scope"].get("start_time"),
            "collection_end_timestamp": config["scope"].get("end_time"),
            "source_raw_data_path": str(raw_dir.resolve()),
            "validated_output_path": str(validated_dir.resolve()),
            "configuration_file": config.get("_config_path", str(DEFAULT_CONFIG_PATH)),
        },
        counts={
            "downloaded_run_rows": len(status_rows),
            "downloaded_raw_files_for_run": sum(bool(row.get("raw_file")) for row in status_rows),
            "shared_raw_directory_files": len(list(raw_dir.glob("*.json"))),
            "unique_analytical_match_ids": len(set(cohort_ids)),
            "duplicate_status_match_ids": len(duplicate_status_ids),
            "raw_checksum_failures": checksum_failures,
            "participant_records": total_participants,
            "patch_17_8_matches": patch_counts["patch_17_8"],
            "non_17_8_matches": patch_counts["non_17_8_patch"],
            "unknown_patch_matches": patch_counts["unknown_patch"],
            "observed_client_patches": sorted({result.get("patch", {}).get("client_patch", "") for result in match_results}),
            "rank_snapshot_rows": len(rank_rows),
            "collection_manifest_present": manifest_path.exists(),
            "collection_status_present": status_path.exists(),
            "accepted_manifest_present": bool(manifest_accepted),
        },
    )
    _write_outputs(report_dir, summary, issues, match_results, missingness_rows,
                   schema_report, rank_result)
    if status != "FAIL":
        _write_validated_dataset(validated_dir, match_results, run_id)
    LOGGER.info("Validation reports written to %s", report_dir)
    return summary, issues


def _rules(config: dict[str, Any]) -> dict[str, Any]:
    validation, scope = config["validation"], config["scope"]
    return {
        "expected_patch": validation.get("expected_patch", scope["study_patch"]),
        "allowed_client_patches": scope["allowed_client_patches"],
        "allowed_queue_ids": scope["allowed_queue_ids"],
        "allowed_game_types": [str(value).lower() for value in scope.get("allowed_game_types", [])],
        "allowed_end_results": scope["allowed_end_results"],
        "expected_participants_per_match": validation.get("expected_participants_per_match", scope["expected_participants"]),
        "valid_placement_min": validation.get("valid_placement_min", 1),
        "valid_placement_max": validation.get("valid_placement_max", 8),
        "sensible_level_min": validation.get("sensible_level_min", 1),
        "sensible_level_max": validation.get("sensible_level_max", 15),
        "sensible_game_duration_max_seconds": validation.get("sensible_game_duration_max_seconds", 14400),
        "fail_on_patch_mismatch": bool(validation.get("fail_on_patch_mismatch", True)),
        "fail_on_end_result_mismatch": bool(validation.get("fail_on_end_result_mismatch", True)),
        "fail_on_duplicate_match_id": bool(validation.get("fail_on_duplicate_match_id", True)),
        "rank_snapshot_required": bool(validation.get("rank_snapshot_required", False)),
    }


def _feature_columns(config: dict[str, Any], run_id: str) -> list[str]:
    path = project_path(config, "processed") / run_id / "features.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle), [])


def _validate_interim_join(config: dict[str, Any], run_id: str,
                           match_results: list[dict[str, Any]]) -> list[Issue]:
    path = project_path(config, "interim") / run_id / "participants.csv"
    if not path.exists():
        return []
    issues: list[Issue] = []
    rows = read_csv_rows(path)
    counts = Counter((row.get("match_hash", ""), row.get("participant_hash", "")) for row in rows)
    for (match_hash, participant_hash), count in counts.items():
        if count > 1:
            issues.append(Issue("critical", "join", "duplicate_participant_join_row",
                                "Derived participant table contains duplicate join rows",
                                match_hash, participant_hash))
    expected_top_four = {
        (result["match_hash"], participant["participant_hash"]): participant["top_four_derived"]
        for result in match_results for participant in result["participants"]
    }
    for row in rows:
        key = (row.get("match_hash", ""), row.get("participant_hash", ""))
        expected = expected_top_four.get(key)
        if expected is None:
            continue
        observed = row.get("top_four", "").strip().lower() == "true"
        if observed != expected:
            issues.append(Issue("critical", "participant", "top_four_derivation_mismatch",
                                "Stored top_four contradicts placement-derived outcome", key[0], key[1]))
    return issues


def _write_outputs(report_dir: Path, summary: ValidationSummary, issues: list[Issue],
                   match_results: list[dict[str, Any]], missingness_rows: list[dict[str, Any]],
                   schema_report: dict[str, Any], rank_result: dict[str, Any]) -> None:
    write_json(report_dir / "validation_summary.json", summary.as_dict())
    flat_summary = {key: value for key, value in summary.as_dict().items() if key not in {"metadata", "counts"}}
    write_csv(report_dir / "validation_summary.csv", [flat_summary], list(flat_summary))
    write_markdown(report_dir / "validation_report.md", summary, issues)
    write_csv(report_dir / "issue_details.csv", [issue.__dict__ for issue in issues], ISSUE_FIELDS)
    write_csv(report_dir / "missingness_summary.csv", missingness_rows,
              ["field", "missing_count", "missing_percentage", "classification"])
    invalid_matches = [{"match_hash": result["match_hash"],
                        "reason_codes": sorted({issue.reason_code for issue in result["issues"] if issue.severity == "critical"})}
                       for result in match_results if not result["valid"]]
    write_csv(report_dir / "invalid_matches.csv", invalid_matches, ["match_hash", "reason_codes"])
    invalid_participants = [issue.__dict__ for issue in issues
                            if issue.scope == "participant" and issue.severity == "critical"]
    write_csv(report_dir / "invalid_participants.csv", invalid_participants, ISSUE_FIELDS)
    patch_rows = [{"match_hash": result["match_hash"], **result["patch"]}
                  for result in match_results if result["patch"].get("classification") != "patch_17_8"]
    write_csv(report_dir / "invalid_patch_matches.csv", patch_rows,
              ["match_hash", "classification", "client_patch", "game_version"])
    end_rows = [{"match_hash": result["match_hash"], "original_end_result": result["end_result_original"],
                 "expected_end_result": "GameComplete", "blank": result["end_result_blank"],
                 "null": result["end_result_null"], "valid": result["end_result_valid"]}
                for result in match_results if not result["end_result_valid"]]
    write_csv(report_dir / "end_result_issues.csv", end_rows,
              ["match_hash", "original_end_result", "expected_end_result", "blank", "null", "valid"])
    write_csv(report_dir / "rank_snapshot_summary.csv", rank_result["summary_rows"],
              list(rank_result["summary_rows"][0]))
    write_csv(report_dir / "unmatched_rank_snapshots.csv", rank_result["unmatched_rows"],
              ["participant_hash", "rank_snapshot_status"])
    write_csv(report_dir / "duplicate_rank_snapshots.csv", rank_result["duplicate_rows"],
              ["participant_hash", "count"])
    duplicate_rows = [issue.__dict__ for issue in issues if "duplicate" in issue.reason_code]
    write_csv(report_dir / "duplicate_records.csv", duplicate_rows, ISSUE_FIELDS)
    exclusion_rows = [{"record_type": issue.scope, "match_hash": issue.match_hash,
                       "participant_hash": issue.participant_hash, "reason_code": issue.reason_code,
                       "reason": issue.reason}
                      for issue in issues if issue.severity == "critical"]
    write_csv(report_dir / "exclusion_log.csv", exclusion_rows,
              ["record_type", "match_hash", "participant_hash", "reason_code", "reason"])
    write_json(report_dir / "schema_report.json", schema_report)


def _write_validated_dataset(validated_dir: Path, match_results: list[dict[str, Any]], run_id: str) -> None:
    matches_dir = validated_dir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for result in match_results:
        if not result["valid"]:
            continue
        sanitized = _sanitize_payload(result["payload"], result)
        output = matches_dir / f"{result['match_hash']}.json"
        write_json(output, sanitized)
        manifest_rows.append({
            "run_id": run_id, "match_hash": result["match_hash"],
            "source_sha256": result["source_sha256"],
            "validated_file": f"matches/{output.name}", "validation_status": "valid",
        })
    write_csv(validated_dir / "validated_manifest.csv", manifest_rows,
              ["run_id", "match_hash", "source_sha256", "validated_file", "validation_status"])


def _sanitize_payload(payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(payload))
    metadata = copied.get("metadata", {})
    info = copied.get("info", {})
    metadata["match_id"] = result["match_hash"]
    metadata["participants"] = [participant["participant_hash"] for participant in result["participants"]]
    for raw_participant, validated in zip(info.get("participants", []), result["participants"]):
        if isinstance(raw_participant, dict):
            raw_participant.pop("riotIdGameName", None)
            raw_participant.pop("riotIdTagline", None)
            raw_participant["participant_hash"] = validated["participant_hash"]
            raw_participant.pop("puuid", None)
            raw_participant["top_four_derived"] = validated["top_four_derived"]
            raw_participant["top_four_reconstruction_flag"] = True
    copied["metadata"] = metadata
    copied["info"] = info
    copied["validation"] = {"match_hash": result["match_hash"], "status": "valid"}
    return copied


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_complete_run(config: dict[str, Any]) -> str:
    candidates: list[tuple[str, str]] = []
    for path in project_path(config, "manifests").glob("collection_run_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "complete":
            candidates.append((str(payload.get("started_at", "")), str(payload.get("run_id", ""))))
    if not candidates:
        raise RuntimeError("No completed collection run was found; supply --run-id to audit a partial run")
    return max(candidates)[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an immutable TFT collection run offline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", default=None,
                        help="Collection run to validate; defaults to the latest completed run")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--validated-dir", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    run_id = args.run_id or _latest_complete_run(config)
    LOGGER.info("Validation start: run=%s raw=%s", run_id, project_path(config, "raw_matches"))
    summary, _ = run_validation(
        config, run_id=run_id,
        report_dir=Path(args.output_dir) if args.output_dir else None,
        validated_dir=Path(args.validated_dir) if args.validated_dir else None,
    )
    print("\nValidation complete\n")
    print(f"Matches checked: {summary.total_matches}")
    print(f"Participants checked: {summary.total_participants}")
    print(f"Patch 17.8: {summary.patch_17_8_percentage:.2f}%")
    print(f"Critical failures: {summary.critical_failures}")
    print(f"Warnings: {summary.warnings}\n")
    print(f"STATUS: {summary.collection_status}")
    if summary.collection_status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

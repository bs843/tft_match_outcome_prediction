from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

from .config import ensure_project_directories, load_config, project_path
from .utils import parse_patch, read_csv_rows, replace_csv_partition, safe_filename, safe_int

QUALITY_FIELDS = [
    "run_id", "match_id", "raw_file", "accepted", "rejection_reasons", "warnings",
    "data_version", "game_version", "study_patch", "client_patch", "set_number", "queue_id", "game_type",
    "end_result", "end_result_source", "game_datetime_ms", "participant_count",
    "invalid_participant_count",
    "missing_puuid_count", "empty_units_count", "missing_traits_count",
    "metadata_participant_mismatch",
]
ID_FIELDS = ["run_id", "match_id"]


@dataclass
class ValidationResult:
    match_id: str
    run_id: str = ""
    raw_file: str = ""
    accepted: bool = True
    rejection_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data_version: str = ""
    game_version: str = ""
    study_patch: str = ""
    client_patch: str = ""
    set_number: int | None = None
    queue_id: int | None = None
    game_type: str = ""
    end_result: str = ""
    end_result_source: str = ""
    game_datetime_ms: int | None = None
    participant_count: int = 0
    invalid_participant_count: int = 0
    missing_puuid_count: int = 0
    empty_units_count: int = 0
    missing_traits_count: int = 0
    metadata_participant_mismatch: bool = False

    def reject(self, reason: str) -> None:
        if reason not in self.rejection_reasons:
            self.rejection_reasons.append(reason)
        self.accepted = False

    def as_row(self) -> dict[str, Any]:
        row = vars(self).copy()
        row["rejection_reasons"] = " | ".join(self.rejection_reasons)
        row["warnings"] = " | ".join(self.warnings)
        return row


def validate_match_payload(
    payload: Any, config: dict[str, Any], *, expected_match_id: str = "",
    raw_file: str = "", run_id: str = "",
) -> ValidationResult:
    result = ValidationResult(expected_match_id, run_id=run_id, raw_file=raw_file)
    scope, validation = config["scope"], config["validation"]
    if not isinstance(payload, dict):
        result.reject("payload_not_object")
        return result
    metadata, info = payload.get("metadata"), payload.get("info")
    if not isinstance(metadata, dict):
        result.reject("missing_metadata")
        metadata = {}
    if not isinstance(info, dict):
        result.reject("missing_info")
        info = {}

    actual_match_id = str(metadata.get("match_id", "")).strip()
    result.match_id = actual_match_id or expected_match_id
    if expected_match_id and actual_match_id != expected_match_id:
        result.reject("match_id_mismatch")
    if not result.match_id:
        result.reject("missing_match_id")
    result.data_version = str(metadata.get("data_version", ""))
    result.game_version = str(info.get("game_version", ""))
    result.study_patch = str(scope["study_patch"])
    result.client_patch = parse_patch(result.game_version)
    result.set_number = safe_int(info.get("tft_set_number"))
    result.queue_id = safe_int(info.get("queue_id"))
    result.game_type = str(info.get("tft_game_type", "")).strip()
    # Observed tft-match-v1 JSON uses camelCase. Reading the snake_case
    # DTO-style name silently bypassed the intended completion gate.
    if "endOfGameResult" in info:
        result.end_result = str(info.get("endOfGameResult", "")).strip()
        result.end_result_source = "info.endOfGameResult"
    result.game_datetime_ms = safe_int(info.get("game_datetime"))

    allowed_client_patches = {str(value).strip() for value in scope["allowed_client_patches"]}
    allowed_queues = {int(value) for value in scope["allowed_queue_ids"]}
    allowed_game_types = {str(value).lower() for value in scope.get("allowed_game_types", [])}
    allowed_end_results = {str(value) for value in scope.get("allowed_end_results", [])}
    _enforce(result, result.client_patch in allowed_client_patches, "wrong_or_missing_client_patch",
             bool(validation["reject_on_wrong_patch"]))
    _enforce(result, result.set_number == int(scope["set_number"]), "wrong_or_missing_set",
             bool(validation["reject_on_wrong_set"]))
    _enforce(result, result.queue_id in allowed_queues, "wrong_or_missing_queue",
             bool(validation["reject_on_wrong_queue"]))
    if allowed_game_types:
        _enforce(result, result.game_type.lower() in allowed_game_types,
                 "wrong_or_missing_game_type", bool(validation.get("reject_on_wrong_game_type", True)))
    if allowed_end_results:
        _enforce(result, result.end_result in allowed_end_results,
                 "wrong_or_missing_end_result", bool(validation.get("reject_on_wrong_end_result", True)))

    if validation.get("reject_outside_collection_window", True):
        start, end = scope.get("start_time"), scope.get("end_time")
        if start is not None and end is not None:
            timestamp_seconds = (result.game_datetime_ms or 0) / 1000
            if not int(start) <= timestamp_seconds <= int(end):
                result.reject("outside_collection_window")

    participants = info.get("participants")
    if not isinstance(participants, list):
        result.reject("participants_not_list")
        participants = []
    result.participant_count = len(participants)
    expected_count = int(scope["expected_participants"])
    _enforce(result, len(participants) == expected_count, "unexpected_participant_count",
             bool(validation["reject_on_unexpected_participant_count"]))

    placements: list[int] = []
    puuids: list[str] = []
    for participant in participants:
        invalid = False
        if not isinstance(participant, dict):
            result.invalid_participant_count += 1
            continue
        puuid = str(participant.get("puuid", "")).strip()
        if not puuid:
            result.missing_puuid_count += 1
            invalid = True
        else:
            puuids.append(puuid)
        placement = safe_int(participant.get("placement"))
        if placement is None or not 1 <= placement <= expected_count:
            invalid = True
        else:
            placements.append(placement)
        level = safe_int(participant.get("level"))
        if level is None or level <= 0:
            invalid = True
        units = participant.get("units")
        if not isinstance(units, list) or not units:
            result.empty_units_count += 1
            invalid = invalid or bool(validation["require_nonempty_units"])
        traits = participant.get("traits")
        if not isinstance(traits, list):
            result.missing_traits_count += 1
            invalid = invalid or bool(validation["require_traits_list"])
        if invalid:
            result.invalid_participant_count += 1

    if len(puuids) != len(set(puuids)):
        result.reject("duplicate_participant_puuid")
    if validation["require_complete_placement_set"] and sorted(placements) != list(range(1, expected_count + 1)):
        result.reject("incomplete_or_duplicate_placements")
    if result.invalid_participant_count:
        _enforce(result, False, "invalid_participant_record",
                 bool(validation["reject_on_invalid_participant"]))
    metadata_puuids = metadata.get("participants")
    if validation.get("require_metadata_participant_match", True):
        if not isinstance(metadata_puuids, list) or set(map(str, metadata_puuids)) != set(puuids):
            result.metadata_participant_mismatch = True
            result.reject("metadata_participant_mismatch")
    return result


def _enforce(result: ValidationResult, condition: bool, reason: str, reject: bool) -> None:
    if condition:
        return
    if reject:
        result.reject(reason)
    elif reason not in result.warnings:
        result.warnings.append(reason)


def validate_run(config: dict[str, Any], run_id: str) -> list[ValidationResult]:
    candidates = [row for row in read_csv_rows(project_path(config, "collection_status"),
                                                ["run_id", "match_id", "raw_file"])
                  if row.get("run_id") == run_id and row.get("raw_file")]
    raw_dir = project_path(config, "raw_matches")
    results: list[ValidationResult] = []
    for row in candidates:
        match_id = row["match_id"]
        raw_path = raw_dir / f"{safe_filename(match_id)}.json"
        try:
            with raw_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            results.append(validate_match_payload(payload, config, expected_match_id=match_id,
                                                   raw_file=row["raw_file"], run_id=run_id))
        except (OSError, json.JSONDecodeError) as exc:
            result = ValidationResult(match_id, run_id=run_id, raw_file=row["raw_file"], accepted=False)
            result.reject(f"unreadable_json:{type(exc).__name__}")
            results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one immutable collection run")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--accepted-target", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    results = validate_run(config, args.run_id)
    status_rows = [
        row for row in read_csv_rows(project_path(config, "collection_status"))
        if row.get("run_id") == args.run_id
    ]
    selected_ids = {
        row["match_id"] for row in status_rows
        if str(row.get("validation_accepted", "")).strip().lower() == "true"
    }
    selected = [result for result in results if result.accepted and result.match_id in selected_ids]
    if args.accepted_target is not None:
        selected = selected[:args.accepted_target]
    replace_csv_partition(project_path(config, "quality_report"),
                          [result.as_row() for result in results], QUALITY_FIELDS,
                          partition_field="run_id", partition_value=args.run_id)
    replace_csv_partition(project_path(config, "accepted_match_ids"),
                           [{"run_id": args.run_id, "match_id": result.match_id}
                            for result in selected], ID_FIELDS,
                          partition_field="run_id", partition_value=args.run_id)
    replace_csv_partition(project_path(config, "rejected_match_ids"),
                          [{"run_id": args.run_id, "match_id": result.match_id}
                           for result in results if not result.accepted], ID_FIELDS,
                          partition_field="run_id", partition_value=args.run_id)
    eligible = sum(result.accepted for result in results)
    print(f"Validated {len(results)} candidates: {eligible} eligible, {len(selected)} selected")
    if not results or not selected:
        raise SystemExit("Validation produced no accepted matches")
    if args.accepted_target is not None and len(selected) != args.accepted_target:
        raise SystemExit(
            f"Validation selected {len(selected)} matches; expected {args.accepted_target}"
        )


if __name__ == "__main__":
    main()

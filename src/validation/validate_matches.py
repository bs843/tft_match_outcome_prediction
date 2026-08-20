from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from ..utils import safe_int
from .validate_participants import validate_participant
from .validate_patch import classify_patch
from .validation_report import Issue


def validate_match(
    payload: Any, *, expected_file_match_id: str, hash_identifier: Callable[[str], str],
    rules: dict[str, Any], seen_match_ids: set[str],
) -> dict[str, Any]:
    issues: list[Issue] = []
    missingness: Counter[str] = Counter()
    if not isinstance(payload, dict):
        return {"issues": [Issue("critical", "match", "payload_not_object",
                                 "Raw payload is not an object")], "participants": [],
                "missingness": missingness, "match_hash": "", "patch": {}}
    metadata = payload.get("metadata")
    info = payload.get("info")
    actual_id = str(metadata.get("match_id", "")).strip() if isinstance(metadata, dict) else ""
    match_hash = hash_identifier(actual_id or expected_file_match_id)
    if not isinstance(metadata, dict):
        issues.append(Issue("critical", "match", "missing_metadata", "Match metadata is missing", match_hash=match_hash))
        metadata = {}
    if not isinstance(info, dict):
        issues.append(Issue("critical", "match", "missing_info", "Match info is missing", match_hash=match_hash))
        info = {}
    if not actual_id:
        issues.append(Issue("critical", "match", "missing_match_id", "metadata.match_id is missing", match_hash=match_hash))
    elif actual_id != expected_file_match_id:
        issues.append(Issue("critical", "match", "match_id_mismatch",
                            "Filename/status match ID disagrees with metadata.match_id", match_hash=match_hash))
    if actual_id in seen_match_ids:
        severity = "critical" if rules.get("fail_on_duplicate_match_id", True) else "warning"
        issues.append(Issue(severity, "match", "duplicate_match_id", "Duplicate metadata.match_id", match_hash=match_hash))
    if actual_id:
        seen_match_ids.add(actual_id)

    patch = classify_patch(payload, set(rules["allowed_client_patches"]))
    if patch["classification"] == "non_17_8_patch":
        severity = "critical" if rules["fail_on_patch_mismatch"] else "warning"
        issues.append(Issue(severity, "patch", "non_17_8_patch", "Client patch is outside Patch 17.8", match_hash=match_hash))
    elif patch["classification"] == "unknown_patch":
        issues.append(Issue("critical", "patch", "unknown_patch", "game_version cannot be parsed", match_hash=match_hash))

    expected_count = int(rules["expected_participants_per_match"])
    participants = info.get("participants")
    if not isinstance(participants, list):
        issues.append(Issue("critical", "match", "participants_not_list", "info.participants is not a list", match_hash=match_hash))
        participants = []
    queue_id = safe_int(info.get("queue_id"))
    standard = queue_id in set(rules["allowed_queue_ids"]) and str(info.get("tft_game_type", "")).lower() in set(rules["allowed_game_types"])
    if len(participants) != expected_count:
        severity = "critical" if standard else "warning"
        issues.append(Issue(severity, "match", "unexpected_participant_count",
                            f"Observed {len(participants)} participants; expected {expected_count}", match_hash=match_hash))

    placements: list[int] = []
    participant_results: list[dict[str, Any]] = []
    seen_participants: set[str] = set()
    for participant in participants:
        raw_identifier = str(participant.get("puuid", "")).strip() if isinstance(participant, dict) else ""
        participant_hash = hash_identifier(raw_identifier) if raw_identifier else ""
        participant_issues, participant_missing, derived = validate_participant(
            participant, match_hash=match_hash, participant_hash=participant_hash,
            placement_min=int(rules["valid_placement_min"]),
            placement_max=int(rules["valid_placement_max"]),
            level_min=int(rules["sensible_level_min"]),
            level_max=int(rules["sensible_level_max"]),
        )
        issues.extend(participant_issues)
        missingness.update(participant_missing)
        if participant_hash and participant_hash in seen_participants:
            issues.append(Issue("critical", "participant", "duplicate_participant",
                                "Duplicate participant identifier within match", match_hash, participant_hash))
        if participant_hash:
            seen_participants.add(participant_hash)
        if derived["placement"] is not None:
            placements.append(derived["placement"])
        participant_results.append({
            "participant_hash": participant_hash,
            "placement": derived["placement"],
            "top_four_derived": derived["top_four"],
            "valid": derived["valid"] and not any(
                issue.severity == "critical" and issue.participant_hash == participant_hash
                for issue in issues
            ),
        })

    expected_placements = list(range(int(rules["valid_placement_min"]), int(rules["valid_placement_max"]) + 1))
    if standard and sorted(placements) != expected_placements:
        issues.append(Issue("critical", "match", "invalid_placement_structure",
                            "Standard match placements are not exactly 1 through 8", match_hash=match_hash))

    end_result = info.get("endOfGameResult")
    end_blank = end_result == ""
    end_null = end_result is None
    end_valid = str(end_result).strip() in set(rules["allowed_end_results"])
    if not end_valid:
        severity = "critical" if rules["fail_on_end_result_mismatch"] else "warning"
        issues.append(Issue(severity, "match", "invalid_end_result",
                            "info.endOfGameResult is blank, null, or invalid", match_hash=match_hash,
                            field="info.endOfGameResult"))

    timestamp = safe_int(info.get("game_datetime"))
    if timestamp is None or timestamp <= 0:
        issues.append(Issue("critical", "match", "invalid_timestamp", "game_datetime is missing or invalid", match_hash=match_hash))
    duration = info.get("game_length")
    try:
        duration_value = float(duration)
    except (TypeError, ValueError):
        duration_value = 0.0
    if duration_value <= 0:
        issues.append(Issue("critical", "match", "invalid_duration", "game_length must be positive", match_hash=match_hash))
    elif duration_value > float(rules["sensible_game_duration_max_seconds"]):
        issues.append(Issue("warning", "match", "suspicious_duration", "game_length exceeds the sensible threshold", match_hash=match_hash))
    if queue_id is None or safe_int(info.get("tft_set_number")) is None:
        issues.append(Issue("critical", "match", "missing_scope_metadata", "queue_id or tft_set_number is missing", match_hash=match_hash))

    critical = any(issue.severity == "critical" for issue in issues)
    return {
        "match_hash": match_hash,
        "match_id": actual_id or expected_file_match_id,
        "issues": issues,
        "participants": participant_results,
        "missingness": missingness,
        "patch": patch,
        "end_result_original": "" if end_result is None else str(end_result),
        "end_result_blank": end_blank,
        "end_result_null": end_null,
        "end_result_valid": end_valid,
        "game_datetime_ms": timestamp,
        "valid": not critical,
        "payload": payload,
    }

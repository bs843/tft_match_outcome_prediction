from __future__ import annotations

from collections import Counter
from typing import Any

from .validation_report import Issue

EXPECTED_SCHEMA = {
    "metadata": {
        "match_id": "identifier", "participants": "identifier", "data_version": "metadata",
    },
    "info": {
        "game_version": "metadata", "queue_id": "metadata", "tft_set_number": "metadata",
        "tft_game_type": "metadata", "game_datetime": "metadata", "game_length": "metadata",
        "endOfGameResult": "required", "participants": "required",
    },
    "participant": {
        "puuid": "identifier", "placement": "target", "level": "required",
        "units": "required", "traits": "required", "augments": "optional",
        "gold_left": "optional", "last_round": "optional", "players_eliminated": "optional",
        "total_damage_to_players": "optional", "time_eliminated": "optional",
    },
}


def validate_schema(payloads: list[dict[str, Any]]) -> tuple[dict[str, Any], list[Issue]]:
    observed = {"metadata": Counter(), "info": Counter(), "participant": Counter()}
    issues: list[Issue] = []
    for payload in payloads:
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        info = payload.get("info", {}) if isinstance(payload, dict) else {}
        if isinstance(metadata, dict):
            observed["metadata"].update(metadata.keys())
        if isinstance(info, dict):
            observed["info"].update(info.keys())
            participants = info.get("participants", [])
            if isinstance(participants, list):
                for participant in participants:
                    if isinstance(participant, dict):
                        observed["participant"].update(participant.keys())
    sections: dict[str, Any] = {}
    for section, expected in EXPECTED_SCHEMA.items():
        observed_fields = set(observed[section])
        expected_fields = set(expected)
        missing_everywhere = sorted(expected_fields.difference(observed_fields))
        extras = sorted(observed_fields.difference(expected_fields))
        sections[section] = {
            "expected_fields": expected,
            "observed_fields": sorted(observed_fields),
            "missing_everywhere": missing_everywhere,
            "extra_fields": extras,
        }
        for field in missing_everywhere:
            classification = expected[field]
            severity = "critical" if classification in {"required", "identifier", "target"} else "warning"
            issues.append(Issue(severity, "schema", "schema_required_field_absent" if severity == "critical" else "schema_optional_field_absent",
                                f"{section}.{field} was absent from every record", field=f"{section}.{field}"))
        if extras:
            issues.append(Issue("warning", "schema", "schema_drift_extra_fields",
                                f"{section} contains additional Riot fields: {', '.join(extras)}"))
    return {"expected_schema": EXPECTED_SCHEMA, "sections": sections}, issues


RANK_FEATURE_BLOCKLIST = {
    "rank_snapshot_tier", "rank_snapshot_division", "rank_snapshot_league_points",
    "rank_snapshot_wins", "rank_snapshot_losses", "rank_snapshot_queue_type",
    "rank_snapshot_retrieved_at", "rank_snapshot_status", "rank_snapshot_error_class",
}


def validate_rank_feature_exclusion(
    config: dict[str, Any], feature_columns: list[str] | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    prohibited = set(config.get("features", {}).get("prohibited_columns", []))
    missing = RANK_FEATURE_BLOCKLIST.difference(prohibited)
    if missing:
        issues.append(Issue("critical", "feature_schema", "rank_fields_not_blocklisted",
                            "Rank fields are missing from prohibited_columns: " + ", ".join(sorted(missing))))
    leaked = {
        column for column in (feature_columns or [])
        if column in RANK_FEATURE_BLOCKLIST or column.startswith("rank_snapshot_")
    }
    if leaked:
        issues.append(Issue("critical", "feature_schema", "rank_feature_leakage",
                            "Rank snapshot columns entered the primary feature matrix: " + ", ".join(sorted(leaked))))
    return issues

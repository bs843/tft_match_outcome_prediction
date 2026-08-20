from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from .validation_report import Issue

VALID_TIERS = {"IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"}
VALID_DIVISIONS = {"I", "II", "III", "IV", ""}
VALID_STATUSES = {"found", "unranked", "not_found", "request_failed"}


def validate_rank_snapshots(
    rows: list[dict[str, str]], *, expected_participants: set[str],
    participant_latest_match_ms: dict[str, int], required: bool,
) -> dict[str, Any]:
    issues: list[Issue] = []
    seen: Counter[str] = Counter(row.get("participant_hash", "") for row in rows)
    duplicate_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    for participant_hash, count in seen.items():
        if participant_hash and count > 1:
            duplicate_rows.append({"participant_hash": participant_hash, "count": count})
            issues.append(Issue("critical", "rank_snapshot", "duplicate_rank_snapshot",
                                "Multiple rank snapshots exist for one participant", participant_hash=participant_hash))
    snapshot_ids = {value for value in seen if value}
    missing_ids = expected_participants.difference(snapshot_ids)
    if missing_ids:
        issues.append(Issue("critical" if required else "warning", "rank_snapshot", "missing_rank_snapshot",
                            f"{len(missing_ids)} accepted participants have no rank snapshot"))
    for row in rows:
        participant_hash = row.get("participant_hash", "")
        if not participant_hash:
            issues.append(Issue("critical", "rank_snapshot", "missing_rank_join_key",
                                "Rank snapshot participant_hash is blank"))
            continue
        if participant_hash not in expected_participants:
            unmatched_rows.append({"participant_hash": participant_hash,
                                   "rank_snapshot_status": row.get("rank_snapshot_status", "")})
            issues.append(Issue("warning", "rank_snapshot", "unmatched_rank_snapshot",
                                "Rank snapshot does not join to an accepted participant",
                                participant_hash=participant_hash))
        status = row.get("rank_snapshot_status", "")
        if status not in VALID_STATUSES:
            issues.append(Issue("critical", "rank_snapshot", "invalid_rank_status",
                                "Rank snapshot status is invalid", participant_hash=participant_hash))
        if status == "found":
            if row.get("rank_snapshot_tier", "").upper() not in VALID_TIERS:
                issues.append(Issue("critical", "rank_snapshot", "invalid_rank_tier",
                                    "Rank tier is invalid", participant_hash=participant_hash))
            if row.get("rank_snapshot_division", "").upper() not in VALID_DIVISIONS:
                issues.append(Issue("critical", "rank_snapshot", "invalid_rank_division",
                                    "Rank division is invalid", participant_hash=participant_hash))
        timestamp = row.get("rank_snapshot_retrieved_at", "")
        try:
            snapshot_ms = int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)
        except (TypeError, ValueError):
            issues.append(Issue("critical", "rank_snapshot", "invalid_rank_timestamp",
                                "Rank snapshot timestamp is missing or unparseable",
                                participant_hash=participant_hash))
            continue
        latest_match = participant_latest_match_ms.get(participant_hash)
        if latest_match and snapshot_ms < latest_match:
            issues.append(Issue("warning", "rank_snapshot", "rank_snapshot_precedes_match",
                                "Rank snapshot predates an accepted match for this participant",
                                participant_hash=participant_hash))
    matched = len(expected_participants.intersection(snapshot_ids))
    return {
        "issues": issues,
        "duplicate_rows": duplicate_rows,
        "unmatched_rows": unmatched_rows,
        "summary_rows": [{
            "expected_unique_participants": len(expected_participants),
            "snapshot_rows": len(rows),
            "matched_unique_participants": matched,
            "missing_unique_participants": len(missing_ids),
            "unmatched_snapshot_rows": len(unmatched_rows),
            "duplicate_snapshot_keys": len(duplicate_rows),
            "match_rate_percentage": round(100 * matched / len(expected_participants), 4) if expected_participants else 0.0,
            "found": sum(row.get("rank_snapshot_status") == "found" for row in rows),
            "unranked": sum(row.get("rank_snapshot_status") == "unranked" for row in rows),
            "not_found": sum(row.get("rank_snapshot_status") == "not_found" for row in rows),
            "request_failed": sum(row.get("rank_snapshot_status") == "request_failed" for row in rows),
        }],
    }

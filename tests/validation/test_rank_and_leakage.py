from __future__ import annotations

from src.validation.validate_rank_snapshots import validate_rank_snapshots
from src.validation.validate_schema import RANK_FEATURE_BLOCKLIST, validate_rank_feature_exclusion


def row(participant_hash: str = "p1") -> dict[str, str]:
    return {
        "participant_hash": participant_hash, "rank_snapshot_status": "found",
        "rank_snapshot_tier": "GOLD", "rank_snapshot_division": "II",
        "rank_snapshot_retrieved_at": "2026-08-12T00:00:00+00:00",
    }


def test_missing_rank_snapshot_is_warning() -> None:
    result = validate_rank_snapshots([], expected_participants={"p1"},
                                     participant_latest_match_ms={}, required=False)
    assert any(issue.reason_code == "missing_rank_snapshot" and issue.severity == "warning"
               for issue in result["issues"])


def test_duplicate_rank_snapshot_is_critical() -> None:
    result = validate_rank_snapshots([row(), row()], expected_participants={"p1"},
                                     participant_latest_match_ms={}, required=False)
    assert any(issue.reason_code == "duplicate_rank_snapshot" and issue.severity == "critical"
               for issue in result["issues"])


def test_rank_feature_leakage_is_critical() -> None:
    config = {"features": {"prohibited_columns": sorted(RANK_FEATURE_BLOCKLIST)}}
    assert validate_rank_feature_exclusion(config, []) == []
    issues = validate_rank_feature_exclusion(config, ["rank_snapshot_tier"])
    assert any(issue.reason_code == "rank_feature_leakage" for issue in issues)

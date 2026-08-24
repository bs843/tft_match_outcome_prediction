from __future__ import annotations

from pathlib import Path

import pytest

from src.provenance import accepted_manifest_checksum
from src.recover_rank_snapshots import (
    RECOVERY_STAGE_MODULES, RankRecoveryError, validate_recovery_authorization,
)
from src.utils import (
    atomic_json_dump, read_csv_rows, sha256_file, stable_json_hash, write_csv_rows,
)


def recovery_fixture(monkeypatch, tmp_path: Path) -> tuple[dict, str]:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "z" * 32)
    run_id = "recoverable-run"
    raw = tmp_path / "raw"
    manifests = tmp_path / "manifests"
    raw.mkdir(); manifests.mkdir()
    accepted_rows = []
    for index, platform in enumerate(("ME1", "EUW1"), start=1):
        match_id = f"{platform}_{index}"
        path = raw / f"{match_id}.json"
        atomic_json_dump({
            "info": {
                "game_datetime": index * 1000,
                "participants": [{"puuid": f"player-{index}"}],
            }
        }, path)
        accepted_rows.append({
            "run_id": run_id, "match_id": match_id,
            "platform_region": platform, "regional_route": "EUROPE",
            "raw_sha256": sha256_file(path), "validation_accepted": True,
        })
    accepted = tmp_path / "accepted.csv"
    fields = ["run_id", "match_id", "platform_region", "regional_route",
              "raw_sha256", "validation_accepted"]
    write_csv_rows(accepted, accepted_rows, fields)
    stored_rows = read_csv_rows(accepted)
    authorization = {
        "amendment_version": "rank_snapshot_routing_v2",
        "authorized_run_id": run_id,
        "original_configuration_version": "old-version",
        "expected_accepted_matches": 2,
        "expected_unique_participants": 2,
        "expected_multi_platform_participants": 0,
        "accepted_manifest_sha256": accepted_manifest_checksum(run_id, stored_rows),
        "accepted_match_id_set_sha256": stable_json_hash(
            sorted(row["match_id"] for row in accepted_rows)
        ),
        "accepted_file_sha256": sha256_file(accepted),
    }
    config = {
        "privacy": {"hash_key_env": "PARTICIPANT_HASH_KEY"},
        "downstream_recovery": authorization,
        "paths": {
            "accepted_match_ids": str(accepted), "raw_matches": str(raw),
            "manifests": str(manifests),
        },
    }
    atomic_json_dump({
        "run_id": run_id, "status": "failed",
        "configuration_version": "old-version",
        "requested_accepted_match_target": 5000,
    }, manifests / f"collection_run_{run_id}.json")
    return config, run_id


def test_recovery_authorization_proves_cohort_and_participant_counts(
    monkeypatch, tmp_path: Path,
) -> None:
    config, run_id = recovery_fixture(monkeypatch, tmp_path)
    _, fingerprint, routing = validate_recovery_authorization(config, run_id)
    assert fingerprint["accepted_match_count"] == 2
    assert fingerprint["raw_checksum_failures"] == 0
    assert len(routing) == 2


def test_recovery_rejects_any_accepted_cohort_change(monkeypatch, tmp_path: Path) -> None:
    config, run_id = recovery_fixture(monkeypatch, tmp_path)
    rows = read_csv_rows(Path(config["paths"]["accepted_match_ids"]))
    rows.pop()
    write_csv_rows(Path(config["paths"]["accepted_match_ids"]), rows, list(rows[0]))
    with pytest.raises(RankRecoveryError, match="integrity gate failed"):
        validate_recovery_authorization(config, run_id)


def test_recovery_stage_allowlist_cannot_collect_candidates_or_matches() -> None:
    assert RECOVERY_STAGE_MODULES == (
        "src.collect_rank_snapshots", "src.normalise_matches",
        "src.create_splits", "src.build_features",
    )
    assert "src.collect_players" not in RECOVERY_STAGE_MODULES
    assert "src.collect_match_ids" not in RECOVERY_STAGE_MODULES
    assert "src.collect_matches" not in RECOVERY_STAGE_MODULES

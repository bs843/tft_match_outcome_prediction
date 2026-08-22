from __future__ import annotations

import csv
import json
from pathlib import Path

from src.utils import atomic_json_dump, hash_identifier, sha256_file, write_csv_rows
from src.validation.validate_collection import _validate_interim_join, run_validation

from .helpers import valid_match


def test_read_only_validation_writes_sanitized_dataset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "v" * 32)
    raw = tmp_path / "raw"; raw.mkdir()
    private = tmp_path / "private"; private.mkdir()
    manifests = tmp_path / "manifests"; manifests.mkdir()
    reports = tmp_path / "reports"; validated = tmp_path / "validated"
    interim = tmp_path / "interim"; processed = tmp_path / "processed"
    match_id = "ME1_TEST"; run_id = "run-1"
    atomic_json_dump(valid_match(match_id), raw / f"{match_id}.json")
    write_csv_rows(private / "status.csv", [{
        "run_id": run_id, "match_id": match_id, "raw_file": "raw/ME1_TEST.json",
        "raw_sha256": sha256_file(raw / f"{match_id}.json"), "validation_accepted": True,
        "platform_region": "ME1", "regional_route": "EUROPE",
    }], ["run_id", "match_id", "raw_file", "raw_sha256", "validation_accepted",
          "platform_region", "regional_route"])
    write_csv_rows(private / "accepted.csv", [{"run_id": run_id, "match_id": match_id,
                                                "platform_region": "ME1", "regional_route": "EUROPE"}],
                   ["run_id", "match_id", "platform_region", "regional_route"])
    key = "v" * 32
    rank_rows = [{
        "run_id": run_id, "participant_hash": hash_identifier(f"private-{p}", key),
        "rank_snapshot_status": "found", "rank_snapshot_tier": "GOLD",
        "rank_snapshot_division": "II", "rank_snapshot_retrieved_at": "2026-08-12T00:00:00+00:00",
    } for p in range(1, 9)]
    write_csv_rows(private / "rank.csv", rank_rows,
                   ["run_id", "participant_hash", "rank_snapshot_status", "rank_snapshot_tier",
                    "rank_snapshot_division", "rank_snapshot_retrieved_at"])
    atomic_json_dump({"run_id": run_id, "status": "complete", "started_at": "2026-08-12T00:00:00Z",
                      "dataset_version": "test", "schema_version": "1"},
                     manifests / f"collection_run_{run_id}.json")
    config = {
        "project": {"dataset_version": "test", "schema_version": "1"},
        "scope": {"study_patch": "17.8", "allowed_client_patches": ["16.15"],
                  "allowed_queue_ids": [1100], "allowed_game_types": ["standard"],
                  "allowed_end_results": ["GameComplete"], "expected_participants": 8,
                  "start_time": 1, "end_time": 2},
        "validation": {"expected_patch": "17.8", "expected_participants_per_match": 8,
                       "valid_placement_min": 1, "valid_placement_max": 8,
                       "sensible_level_min": 1, "sensible_level_max": 15,
                       "sensible_game_duration_max_seconds": 14400,
                       "fail_on_patch_mismatch": True, "fail_on_end_result_mismatch": True,
                       "rank_snapshot_required": False},
        "features": {"prohibited_columns": [
            "rank_snapshot_tier", "rank_snapshot_division", "rank_snapshot_league_points",
            "rank_snapshot_wins", "rank_snapshot_losses", "rank_snapshot_queue_type",
            "rank_snapshot_retrieved_at", "rank_snapshot_status", "rank_snapshot_error_class",
        ]},
        "privacy": {"hash_key_env": "PARTICIPANT_HASH_KEY"},
        "paths": {
            "raw_matches": str(raw), "manifests": str(manifests),
            "collection_status": str(private / "status.csv"),
            "accepted_match_ids": str(private / "accepted.csv"),
            "rank_snapshots": str(private / "rank.csv"), "processed": str(processed),
            "interim": str(interim), "validation_reports": str(reports),
            "validated": str(validated),
        },
        "_config_path": str(tmp_path / "config.yaml"),
    }
    before = sha256_file(raw / f"{match_id}.json")
    summary, _ = run_validation(config, run_id=run_id)
    assert summary.collection_status == "PASS"
    assert sha256_file(raw / f"{match_id}.json") == before
    output = next((validated / run_id / "matches").glob("*.json"))
    text = output.read_text(encoding="utf-8")
    assert "private-" not in text
    assert "participant_hash" in text

    # The same accepted partition cannot become analytical output if its run
    # manifest is failed/interrupted.
    atomic_json_dump({"run_id": run_id, "status": "failed", "started_at": "2026-08-12T00:00:00Z",
                      "dataset_version": "test", "schema_version": "1"},
                     manifests / f"collection_run_{run_id}.json")
    failed_validated = tmp_path / "failed-validated"
    failed_summary, _ = run_validation(
        config, run_id=run_id, report_dir=tmp_path / "failed-report",
        validated_dir=failed_validated,
    )
    assert failed_summary.collection_status == "FAIL"
    assert not failed_validated.exists()


def test_placement_top_four_mismatch_and_duplicate_join_are_critical(tmp_path: Path) -> None:
    run = "run"; path = tmp_path / run; path.mkdir()
    fields = ["match_hash", "participant_hash", "placement", "top_four"]
    write_csv_rows(path / "participants.csv", [
        {"match_hash": "m", "participant_hash": "p", "placement": 1, "top_four": False},
        {"match_hash": "m", "participant_hash": "p", "placement": 1, "top_four": False},
    ], fields)
    config = {"paths": {"interim": str(tmp_path)}}
    results = [{"match_hash": "m", "participants": [{
        "participant_hash": "p", "top_four_derived": True,
    }]}]
    issues = _validate_interim_join(config, run, results)
    codes = {issue.reason_code for issue in issues}
    assert {"duplicate_participant_join_row", "top_four_derivation_mismatch"}.issubset(codes)

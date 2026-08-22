from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import analysis.src.gate as gate_module
from analysis.src.gate import AnalysisGateError, require_analysis_approval
from src.config import load_config
from src.provenance import accepted_manifest_checksum, config_hash


RUN_ID = "fixture-run"


def test_gate_fails_closed_for_missing_manifest(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(gate_module, "PROJECT_ROOT", tmp_path)
    with pytest.raises(AnalysisGateError, match="collection manifest is missing"):
        require_analysis_approval(config, RUN_ID)


def test_gate_approves_only_complete_consistent_cohort(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(gate_module, "PROJECT_ROOT", tmp_path)
    _approved_fixture(tmp_path, config)
    approval = require_analysis_approval(config, RUN_ID)
    assert approval.accepted_count == 2
    assert len(approval.validated_files) == 2
    assert all("validated" in str(path) for path in approval.validated_files)


def test_gate_rejects_validation_error(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(gate_module, "PROJECT_ROOT", tmp_path)
    _approved_fixture(tmp_path, config)
    report_path = tmp_path / "reports/validation" / RUN_ID / "validation_summary.json"
    report = json.loads(report_path.read_text())
    report["invalid_matches"] = 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(AnalysisGateError, match="invalid_matches"):
        require_analysis_approval(config, RUN_ID)


def _config(tmp_path: Path) -> dict:
    config = load_config()
    config["sampling"]["target_accepted_matches"] = 2
    config["scope"]["expected_participants"] = 8
    config["_config_path"] = str(tmp_path / "config.yaml")
    return config


def _approved_fixture(root: Path, config: dict) -> None:
    accepted_path = root / config["paths"]["accepted_match_ids"]
    accepted_path.parent.mkdir(parents=True)
    accepted = [{"run_id": RUN_ID, "match_id": "raw-one", "platform_region": "ME1",
                 "regional_route": "EUROPE", "raw_sha256": "a" * 64,
                 "validation_accepted": "true"},
                {"run_id": RUN_ID, "match_id": "raw-two", "platform_region": "ME1",
                 "regional_route": "EUROPE", "raw_sha256": "b" * 64,
                 "validation_accepted": "true"}]
    _csv(accepted_path, accepted)
    checksum = accepted_manifest_checksum(RUN_ID, accepted)
    manifest_path = root / config["paths"]["manifests"] / f"collection_run_{RUN_ID}.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({
        "run_id": RUN_ID, "status": "complete", "requested_accepted_match_target": 2,
        "dataset_version": config["project"]["dataset_version"],
        "schema_version": config["project"]["schema_version"], "study_patch": "17.8",
        "config_hash": config_hash(config), "accepted_match_manifest_id": "fixture",
        "accepted_match_manifest_sha256": checksum, "stage_counts": {"accepted_matches": 2},
        "git_commit": "abc", "git_tag": "tag",
        "enabled_platforms": [{"platform_region": row["platform_region"],
                               "regional_route": row["regional_route"]}
                              for row in config["platforms"] if row["enabled"]],
    }), encoding="utf-8")
    report_path = root / config["paths"]["validation_reports"] / RUN_ID / "validation_summary.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps({
        "collection_status": "PASS", "critical_failures": 0, "invalid_matches": 0,
        "invalid_participants": 0, "duplicate_match_count": 0,
        "missing_required_field_count": 0, "end_result_blank_count": 0,
        "end_result_mismatch_count": 0, "schema_errors": 0,
        "patch_17_8_percentage": 100.0, "valid_matches": 2, "valid_participants": 16,
        "rank_snapshot_match_rate": 0.0,
        "counts": {"raw_checksum_failures": 0, "non_17_8_matches": 0,
                   "unknown_patch_matches": 0, "accepted_manifest_present": True},
        "metadata": {"collection_id": RUN_ID, "collection_manifest_status": "complete",
                     "expected_patch": "17.8", "schema_version": config["project"]["schema_version"],
                     "enabled_platforms": [{"platform_region": row["platform_region"],
                                            "regional_route": row["regional_route"]}
                                           for row in config["platforms"] if row["enabled"]]},
    }), encoding="utf-8")
    validated = root / config["paths"]["validated"] / RUN_ID
    (validated / "matches").mkdir(parents=True)
    rows = []
    for number in range(2):
        file = validated / "matches" / f"hash-{number}.json"
        file.write_text("{}", encoding="utf-8")
        rows.append({"run_id": RUN_ID, "match_hash": f"hash-{number}", "source_sha256": "a" * 64,
                     "validated_file": f"matches/hash-{number}.json", "validation_status": "valid"})
    _csv(validated / "validated_manifest.csv", rows)


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

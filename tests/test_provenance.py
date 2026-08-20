from __future__ import annotations

import subprocess
import json
from pathlib import Path

import pytest

import src.provenance as provenance


TAG = "tft-patch17.8-final-collection-v4"


def git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=path, check=True, capture_output=True, text=True,
    ).stdout.strip()


def repository(tmp_path: Path) -> Path:
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "tests@example.invalid")
    git(tmp_path, "config", "user.name", "Test Suite")
    (tmp_path / "config.yaml").write_text("frozen: true\n", encoding="utf-8")
    git(tmp_path, "add", "config.yaml")
    git(tmp_path, "commit", "-m", "freeze")
    git(tmp_path, "tag", "-a", TAG, "-m", "frozen")
    return tmp_path


def config() -> dict:
    return {"provenance": {"required_git_tag": TAG}}


def test_frozen_git_state_accepts_clean_tagged_head(monkeypatch, tmp_path: Path) -> None:
    repo = repository(tmp_path)
    monkeypatch.setattr(provenance, "PROJECT_ROOT", repo)
    state = provenance.verify_frozen_git_state(config())
    assert state == {"git_commit": git(repo, "rev-parse", "HEAD"), "git_tag": TAG}
    assert len(state["git_commit"]) == 40


def test_frozen_git_state_rejects_tracked_changes(monkeypatch, tmp_path: Path) -> None:
    repo = repository(tmp_path)
    monkeypatch.setattr(provenance, "PROJECT_ROOT", repo)
    (repo / "config.yaml").write_text("frozen: false\n", encoding="utf-8")
    with pytest.raises(provenance.GitIntegrityError, match="uncommitted changes"):
        provenance.verify_frozen_git_state(config())


def test_frozen_git_state_rejects_head_after_tag(monkeypatch, tmp_path: Path) -> None:
    repo = repository(tmp_path)
    monkeypatch.setattr(provenance, "PROJECT_ROOT", repo)
    (repo / "README.md").write_text("later change\n", encoding="utf-8")
    git(repo, "add", "README.md")
    git(repo, "commit", "-m", "later")
    with pytest.raises(provenance.GitIntegrityError, match="does not match"):
        provenance.verify_frozen_git_state(config())


def test_accepted_manifest_checksum_is_order_independent() -> None:
    rows = [
        {"run_id": "run", "match_id": "B"},
        {"run_id": "other", "match_id": "ignored"},
        {"run_id": "run", "match_id": "A"},
    ]
    assert provenance.accepted_manifest_checksum("run", rows) == provenance.accepted_manifest_checksum(
        "run", list(reversed(rows))
    )


def test_collection_manifest_records_v4_git_and_configuration_provenance(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    config_value = {
        "project": {"dataset_version": "dataset-v5", "schema_version": "1.1.0"},
        "provenance": {"configuration_version": TAG},
        "scope": {"study_patch": "17.8"},
        "paths": {"manifests": str(manifests)},
    }
    path = provenance.initialise_run_manifest(
        config_value, "new-run",
        git_state={"git_commit": "a" * 40, "git_tag": TAG},
        requested_accepted_match_target=5000,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["git_commit"] == "a" * 40
    assert payload["git_tag"] == TAG
    assert payload["configuration_version"] == TAG
    assert payload["requested_accepted_match_target"] == 5000
    provenance.update_run_manifest(path, status="capacity_failed")
    failed = json.loads(path.read_text(encoding="utf-8"))
    assert failed["status"] == "capacity_failed"
    assert failed["completed_at"]
    assert failed["accepted_match_manifest_id"] is None
    assert failed["accepted_match_manifest_sha256"] is None

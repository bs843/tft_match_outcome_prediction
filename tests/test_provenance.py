from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import src.provenance as provenance


TAG = "tft-patch17.8-final-collection-v2"


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

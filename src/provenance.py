from __future__ import annotations

import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, project_path
from .utils import atomic_json_dump, stable_json_hash, utc_now_iso


class GitIntegrityError(RuntimeError):
    """Raised when a final collection is not running from its frozen Git tag."""


def _git(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise GitIntegrityError(f"Git provenance check failed: {detail.strip()}") from exc
    return completed.stdout.strip()


def new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_hash(config: dict[str, Any]) -> str:
    return stable_json_hash(public_config(config))


def git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True, timeout=5,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def verify_frozen_git_state(config: dict[str, Any]) -> dict[str, str]:
    """Require final collection to run from a clean, immutable annotated tag.

    Untracked files are ignored deliberately: private raw/runtime artifacts are
    excluded by .gitignore and must not make a resumed command appear dirty.
    Changes to any tracked file abort the collection.
    """
    tag = str(config.get("provenance", {}).get("required_git_tag", "")).strip()
    if not tag:
        raise GitIntegrityError("No required Git collection tag is configured")
    head = _git("rev-parse", "HEAD")
    tagged_commit = _git("rev-parse", f"{tag}^{{commit}}")
    if head != tagged_commit:
        raise GitIntegrityError(
            f"HEAD {head} does not match frozen collection tag {tag} ({tagged_commit})"
        )
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise GitIntegrityError(
            "Tracked files contain uncommitted changes; review, commit, push, and "
            "create a new version/tag before final collection"
        )
    return {"git_commit": head, "git_tag": tag}


def accepted_manifest_checksum(run_id: str, rows: list[dict[str, str]]) -> str:
    partition = sorted(
        ({"run_id": run_id, "match_id": row.get("match_id", "")} for row in rows
         if row.get("run_id") == run_id),
        key=lambda row: row["match_id"],
    )
    return stable_json_hash(partition)


def initialise_run_manifest(
    config: dict[str, Any], run_id: str, *, git_state: dict[str, str] | None = None,
    requested_accepted_match_target: int | None = None,
) -> Path:
    path = project_path(config, "manifests") / f"collection_run_{run_id}.json"
    if path.exists():
        return path
    atomic_json_dump(
        {
            "run_id": run_id,
            "status": "started",
            "started_at": utc_now_iso(),
            "completed_at": None,
            "config_hash": config_hash(config),
            "dataset_version": config["project"]["dataset_version"],
            "schema_version": config["project"]["schema_version"],
            "git_commit": (git_state or {}).get("git_commit", git_commit()),
            "git_tag": (git_state or {}).get("git_tag", "unavailable"),
            "study_patch": config["scope"]["study_patch"],
            "requested_accepted_match_target": requested_accepted_match_target,
            "accepted_match_manifest_id": None,
            "accepted_match_manifest_sha256": None,
            "stage_counts": {},
        },
        path,
    )
    return path


def update_run_manifest(
    path: Path,
    *,
    status: str,
    stage_counts: dict[str, int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    from .utils import load_json

    payload = load_json(path)
    payload["status"] = status
    if stage_counts:
        payload.setdefault("stage_counts", {}).update(stage_counts)
    if metadata:
        payload.update(metadata)
    if status in {"complete", "failed", "partial"}:
        payload["completed_at"] = utc_now_iso()
    atomic_json_dump(payload, path)

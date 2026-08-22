from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from analysis import __version__

from .common import sha256_file, utc_now, write_csv, write_json


class AnalysisCodeStateError(RuntimeError):
    pass


def require_versioned_clean_analysis_code(project_root: Path) -> str:
    """Require production analysis to be attributable to a committed code state."""
    paths = ["analysis", "requirements.txt", "requirements.lock", "pyproject.toml"]
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
            cwd=project_root, check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise AnalysisCodeStateError(f"Cannot establish analysis Git provenance: {exc}") from exc
    if status:
        raise AnalysisCodeStateError(
            "Production analysis requires committed analysis code and dependencies; "
            "commit these files before running the approved dataset"
        )
    return commit


def write_provenance(
    output_dir: Path,
    *,
    analysis_run_id: str,
    mode: str,
    collection: dict[str, Any],
    config_checksums: dict[str, str],
    split_checksum: str,
) -> None:
    payload = {
        "analysis_run_id": analysis_run_id,
        "analysis_mode": mode,
        "created_at_utc": utc_now(),
        "analysis_code_version": __version__,
        "analysis_git_commit": _git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "collection": collection,
        "configuration_checksums": config_checksums,
        "split_checksum": split_checksum,
        "reproduction_command": (
            f"python -m analysis.run --validation-run {collection.get('run_id')}"
            if mode == "PRODUCTION" else "python -m analysis.run --dry-run"
        ),
    }
    write_json(output_dir / "provenance.json", payload)
    _write_artifact_manifest(output_dir)


def write_artifact_sidecar(path: Path, *, artifact_type: str, collection_run_id: str,
                           row_count: int | None = None, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "artifact": path.name,
        "artifact_type": artifact_type,
        "collection_run_id": collection_run_id,
        "created_at_utc": utc_now(),
        "sha256": sha256_file(path),
    }
    if row_count is not None:
        payload["row_count"] = row_count
    if extra:
        payload.update(extra)
    write_json(path.with_suffix(path.suffix + ".provenance.json"), payload)


def _write_artifact_manifest(output_dir: Path) -> None:
    destination = output_dir / "artifact_manifest.csv"
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == destination:
            continue
        rows.append({"relative_path": path.relative_to(output_dir).as_posix(),
                     "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv(destination, rows, ["relative_path", "bytes", "sha256"])


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"

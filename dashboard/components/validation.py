"""Safety checks for the dashboard's presentation-only data boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


class DashboardDataError(RuntimeError):
    """Base class for safe dashboard failures."""


class MissingArtifactError(DashboardDataError):
    """Raised when a required frozen artifact is unavailable."""


class MetricScopeError(DashboardDataError):
    """Raised when validation and final-test results are mixed."""


class ProhibitedColumnsError(DashboardDataError):
    """Raised when a display frame contains identifiers or secrets."""


class ArtifactIntegrityError(DashboardDataError):
    """Raised when a frozen artifact no longer matches its manifest."""


PROHIBITED_EXACT_COLUMNS = {
    "api_key",
    "match_hash",
    "match_id",
    "participant_hash",
    "participant_id",
    "puuid",
    "riot_id",
    "salt",
    "source_sha256",
}

PROHIBITED_COLUMN_FRAGMENTS = (
    "api_key",
    "match_hash",
    "match_id",
    "participant_hash",
    "participant_id",
    "puuid",
    "riot_id",
    "secret",
)


def require_file(path: Path) -> Path:
    """Return an existing file or raise a safe, user-facing exception."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise MissingArtifactError(f"Frozen artifact not found: {path}")
    return resolved


def assert_safe_columns(columns: Iterable[object]) -> None:
    """Reject identifiers and secret-bearing columns before display."""
    normalized = {str(column).strip().lower() for column in columns}
    unsafe = {
        column
        for column in normalized
        if column in PROHIBITED_EXACT_COLUMNS
        or any(fragment in column for fragment in PROHIBITED_COLUMN_FRAGMENTS)
    }
    if unsafe:
        names = ", ".join(sorted(unsafe))
        raise ProhibitedColumnsError(
            f"Dashboard display blocked prohibited column(s): {names}"
        )


def assert_metric_scope(
    frame: pd.DataFrame,
    expected_partition: str,
    partition_column: str = "partition",
) -> None:
    """Ensure a metric table belongs to exactly one declared partition."""
    if partition_column not in frame.columns:
        raise MetricScopeError(
            f"Metric table is missing required scope column: {partition_column}"
        )
    actual = {
        str(value).strip().lower()
        for value in frame[partition_column].dropna().unique()
    }
    expected = expected_partition.strip().lower()
    if actual != {expected}:
        raise MetricScopeError(
            f"Expected only {expected!r} metrics, found {sorted(actual)!r}."
        )


def sha256_file(path: Path) -> str:
    """Hash a file without modifying it."""
    digest = hashlib.sha256()
    with require_file(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_generated_manifest(manifest_path: Path, base_dir: Path) -> dict[str, object]:
    """Verify every file declared by a frozen generated-files manifest."""
    manifest = json.loads(require_file(manifest_path).read_text(encoding="utf-8"))
    failures: list[str] = []
    verified = 0
    base = base_dir.resolve()

    for entry in manifest.get("generated_files", []):
        relative = Path(str(entry["path"]))
        candidate = (base / relative).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            failures.append(f"Unsafe manifest path: {relative}")
            continue
        if not candidate.is_file():
            failures.append(f"Missing: {relative.as_posix()}")
            continue
        actual = sha256_file(candidate)
        if actual != entry.get("sha256"):
            failures.append(f"Hash mismatch: {relative.as_posix()}")
            continue
        verified += 1

    return {
        "status": "PASS" if not failures else "FAIL",
        "verified": verified,
        "expected": len(manifest.get("generated_files", [])),
        "failures": failures,
    }


def optional_file(path: Path) -> Path | None:
    """Resolve an optional artifact without inventing fallback analysis."""
    return path.resolve() if path.is_file() else None


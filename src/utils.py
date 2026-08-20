from __future__ import annotations

import csv
import hashlib
import hmac
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import ConfigurationError

PATCH_PATTERN = re.compile(r"(?:^|\s)(\d{1,2}\.\d{1,2})(?:\.\d+)*(?:\s|$)")
SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def epoch_ms_to_iso(value: Any) -> str:
    try:
        timestamp = float(value) / 1000.0
        return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def get_hash_key(config: dict[str, Any]) -> bytes:
    env_name = config.get("privacy", {}).get("hash_key_env")
    if not env_name:
        raise ConfigurationError("privacy.hash_key_env is not configured")
    value = os.getenv(str(env_name))
    if not value:
        raise ConfigurationError(f"Environment variable {env_name} is missing")
    encoded = value.encode("utf-8")
    if len(encoded) < 32:
        raise ConfigurationError(f"Environment variable {env_name} must be at least 32 bytes")
    return encoded


def hash_identifier(identifier: str, key: bytes | str) -> str:
    """Return a project-specific HMAC pseudonym, never an unkeyed public hash."""
    if not identifier:
        return ""
    encoded_key = key.encode("utf-8") if isinstance(key, str) else key
    return hmac.new(encoded_key, identifier.encode("utf-8"), hashlib.sha256).hexdigest()


def parse_patch(game_version: Any) -> str:
    if game_version is None:
        return ""
    match = PATCH_PATTERN.search(f" {str(game_version).strip()} ")
    return match.group(1) if match else ""


def safe_filename(value: str) -> str:
    safe = SAFE_FILENAME_PATTERN.sub("_", value.strip())
    if not safe:
        raise ValueError("Cannot create a filename from an empty identifier")
    return safe


def placement_group(placement: Any) -> str:
    value = safe_int(placement)
    if value is None or not 1 <= value <= 8:
        return ""
    lower = value if value % 2 else value - 1
    return f"{lower}-{lower + 1}"


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    _replace_with_retry(temporary_path, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_csv_rows(path: Path, required_fields: Iterable[str] = ()) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = set(required_fields).difference(fields)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def write_csv_rows(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialise_csv_value(row.get(key)) for key in fieldnames})
    _replace_with_retry(temporary_path, path)


def replace_csv_partition(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: list[str],
    *,
    partition_field: str,
    partition_value: str,
) -> None:
    """Replace one run partition while preserving all other recorded runs."""
    existing = read_csv_rows(path) if path.exists() else []
    retained = [row for row in existing if row.get(partition_field) != partition_value]
    write_csv_rows(path, [*retained, *list(rows)], fieldnames)


def _replace_with_retry(source: Path, destination: Path, attempts: int = 8) -> None:
    """Retry transient Windows locks caused by scanners or spreadsheet previews."""
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except PermissionError as exc:
            if attempt == attempts - 1:
                raise PermissionError(
                    f"Could not replace {destination}. Close Excel, preview panes, or other "
                    "programs holding the CSV open, then resume the same run."
                ) from exc
            time.sleep(min(2.0, 0.1 * (2 ** attempt)))


def append_csv_row(path: Path, row: Mapping[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: _serialise_csv_value(row.get(key)) for key in fieldnames})


def _serialise_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value

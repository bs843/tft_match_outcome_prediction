from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT
from src.utils import read_csv_rows

from .gate import AnalysisApproval


class DatasetIntegrityError(RuntimeError):
    pass


def build_participant_dataset(
    approval: AnalysisApproval,
    config: dict[str, Any],
    *,
    include_descriptive_rank: bool = True,
    match_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Build one final-board row per participant from approved sanitized JSON only."""
    rank = _load_clean_rank(config, approval.collection_run_id) if include_descriptive_rank else {}
    rows: list[dict[str, Any]] = []
    seen_files: set[Path] = set()
    paths = (approval.validated_files if match_ids is None else
             tuple(approval.validated_file_map[match_id] for match_id in sorted(match_ids)))
    for path in paths:
        if path in seen_files:
            raise DatasetIntegrityError(f"Duplicate validated file reference: {path.name}")
        seen_files.add(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(participant_rows_from_payload(
            payload, approval.collection_run_id, rank,
            study_patch=str(config["scope"]["study_patch"]),
        ))
    frame = pd.DataFrame(rows)
    validate_participant_dataset(frame, approval if match_ids is None else None)
    return frame.sort_values(["match_hash", "participant_slot"], kind="stable").reset_index(drop=True)


def validate_participant_dataset(frame: pd.DataFrame, approval: AnalysisApproval | None = None) -> None:
    required = {
        "match_hash", "participant_hash", "participant_slot", "placement", "top4",
        "level", "units_json", "traits_json", "items_json", "study_patch",
        "set_number", "queue_id",
        "platform_region", "regional_route",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise DatasetIntegrityError(f"Participant dataset is missing: {sorted(missing)}")
    if frame.empty:
        raise DatasetIntegrityError("Participant dataset is empty")
    if frame[["match_hash", "participant_hash"]].duplicated().any():
        raise DatasetIntegrityError("Duplicate match-participant rows detected")
    if frame[["match_hash", "participant_slot"]].duplicated().any():
        raise DatasetIntegrityError("Duplicate match-slot rows detected")
    if set(frame["study_patch"].astype(str)) != {"17.8"}:
        raise DatasetIntegrityError("Unexpected study patch in analytical rows")
    if set(pd.to_numeric(frame["set_number"], errors="coerce")) != {17.0}:
        raise DatasetIntegrityError("Unexpected TFT set in analytical rows")
    if set(pd.to_numeric(frame["queue_id"], errors="coerce")) != {1100.0}:
        raise DatasetIntegrityError("Unexpected queue in analytical rows")
    if frame["platform_region"].astype(str).str.strip().eq("").any():
        raise DatasetIntegrityError("Analytical rows lack platform-region provenance")
    if not (frame.groupby("match_hash")["platform_region"].nunique() == 1).all():
        raise DatasetIntegrityError("A match cannot span multiple platform regions")
    placements = pd.to_numeric(frame["placement"], errors="coerce")
    if placements.isna().any() or not placements.between(1, 8).all():
        raise DatasetIntegrityError("Placement must be an integer from 1 to 8")
    expected_top4 = (placements <= 4).astype(int)
    observed_top4 = pd.to_numeric(frame["top4"], errors="coerce")
    if observed_top4.isna().any() or not observed_top4.equals(expected_top4):
        raise DatasetIntegrityError("top4 must be derived exclusively from placement <= 4")
    per_match = frame.groupby("match_hash", sort=False)
    if not (per_match.size() == 8).all():
        raise DatasetIntegrityError("Every match must contain exactly eight participant rows")
    if not per_match["placement"].apply(lambda x: set(map(int, x)) == set(range(1, 9))).all():
        raise DatasetIntegrityError("Every match must contain placements 1 through 8 exactly once")
    if approval is not None:
        if frame["match_hash"].nunique() != approval.accepted_count:
            raise DatasetIntegrityError("Dataset match count differs from approved cohort")
        if len(frame) != approval.participant_count:
            raise DatasetIntegrityError("Dataset participant count differs from approved cohort")


def participant_rows_from_payload(payload: dict[str, Any], run_id: str,
                                  rank: dict[str, dict[str, Any]], *,
                                  study_patch: str) -> list[dict[str, Any]]:
    metadata = payload.get("metadata")
    info = payload.get("info")
    validation = payload.get("validation")
    if not isinstance(metadata, dict) or not isinstance(info, dict) or not isinstance(validation, dict):
        raise DatasetIntegrityError("Sanitized payload lacks metadata/info/validation objects")
    if validation.get("status") != "valid":
        raise DatasetIntegrityError("Sanitized payload is not marked valid")
    match_hash = str(metadata.get("match_id") or validation.get("match_hash") or "")
    if not match_hash or match_hash != validation.get("match_hash"):
        raise DatasetIntegrityError("Sanitized match hash is absent or inconsistent")
    participants = info.get("participants")
    platform_region = str(validation.get("platform_region") or "").upper()
    regional_route = str(validation.get("regional_route") or "").upper()
    if not platform_region or not regional_route:
        raise DatasetIntegrityError("Sanitized match lacks platform routing provenance")
    if not isinstance(participants, list) or len(participants) != 8:
        raise DatasetIntegrityError("Sanitized match does not contain eight participants")
    result: list[dict[str, Any]] = []
    for slot, participant in enumerate(participants):
        if not isinstance(participant, dict):
            raise DatasetIntegrityError("Participant is not an object")
        participant_hash = str(participant.get("participant_hash") or "")
        placement = _integer(participant.get("placement"), "placement")
        units = _units(participant.get("units"))
        traits = _traits(participant.get("traits"))
        items = [item for unit in units for item in unit["items"]]
        rank_values = rank.get(participant_hash, {})
        result.append({
            "collection_run_id": run_id,
            "match_hash": match_hash,
            "participant_hash": participant_hash,
            "participant_slot": slot,
            "game_datetime": info.get("game_datetime", ""),
            "study_patch": study_patch,
            "set_number": _integer(info.get("tft_set_number"), "tft_set_number"),
            "queue_id": _integer(info.get("queue_id"), "queue_id"),
            "platform_region": platform_region,
            "regional_route": regional_route,
            "placement": placement,
            "top4": int(placement <= 4),
            "level": _integer(participant.get("level"), "level"),
            "composition_size": len(units),
            "units_json": json.dumps(units, sort_keys=True, separators=(",", ":")),
            "traits_json": json.dumps(traits, sort_keys=True, separators=(",", ":")),
            "items_json": json.dumps(items, sort_keys=True, separators=(",", ":")),
            "rank_tier": rank_values.get("rank_tier", ""),
            "rank_division": rank_values.get("rank_division", ""),
            "rank_status": rank_values.get("rank_status", "not_retrieved"),
        })
    return result


def _units(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise DatasetIntegrityError("Participant units must be a list")
    result = []
    for unit in value:
        if not isinstance(unit, dict):
            raise DatasetIntegrityError("Unit entry must be an object")
        name = str(unit.get("character_id") or "").strip()
        if not name:
            raise DatasetIntegrityError("Unit character_id is missing")
        stars = _integer(unit.get("tier"), "unit tier")
        raw_items = unit.get("itemNames", unit.get("items", []))
        if not isinstance(raw_items, list):
            raise DatasetIntegrityError("Unit items must be a list")
        result.append({"name": name, "stars": stars,
                       "items": [str(item) for item in raw_items if str(item).strip()]})
    return result


def _traits(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise DatasetIntegrityError("Participant traits must be a list")
    result = []
    for trait in value:
        if not isinstance(trait, dict):
            raise DatasetIntegrityError("Trait entry must be an object")
        name = str(trait.get("name") or "").strip()
        if not name:
            continue
        result.append({
            "name": name,
            "num_units": _integer(trait.get("num_units", 0), "trait num_units"),
            "style": _integer(trait.get("style", 0), "trait style"),
            "tier_current": _integer(trait.get("tier_current", 0), "trait tier_current"),
        })
    return result


def _load_clean_rank(config: dict[str, Any], run_id: str) -> dict[str, dict[str, Any]]:
    path = PROJECT_ROOT / config["paths"]["rank_snapshots"]
    if not path.exists():
        return {}
    rows = [row for row in read_csv_rows(path) if row.get("run_id") == run_id]
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("participant_hash", "")
        if not key or key in result:
            continue
        status = row.get("rank_snapshot_status", "not_retrieved")
        result[key] = {
            # Descriptive only. Snapshot timestamps, queue bookkeeping, LP and
            # request errors intentionally never enter the participant dataset.
            "rank_tier": row.get("rank_snapshot_tier", "") if status == "found" else "",
            "rank_division": row.get("rank_snapshot_division", "") if status == "found" else "",
            "rank_status": status,
        }
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise DatasetIntegrityError(f"{label} is not an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DatasetIntegrityError(f"{label} is not an integer") from exc

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from typing import Any

from .config import ensure_project_directories, load_config, project_path
from .provenance import config_hash
from .utils import (
    atomic_json_dump, epoch_ms_to_iso, get_hash_key, hash_identifier, load_json, placement_group,
    read_csv_rows, safe_filename, safe_int, sha256_file, stable_json_hash, write_csv_rows,
)

PROVENANCE_FIELDS = ["run_id", "dataset_version", "schema_version"]
MATCH_FIELDS = [
    "match_hash", "game_datetime_ms", "game_datetime_utc", "game_length_seconds",
    "patch", "queue_id", "tft_game_type", "tft_set_number", "participant_count",
] + PROVENANCE_FIELDS
PARTICIPANT_FIELDS = [
    "match_hash", "participant_hash", "placement", "top_four", "placement_group",
    "level", "gold_left", "last_round", "players_eliminated",
    "total_damage_to_players", "time_eliminated", "units_count", "traits_count",
    "active_traits_count", "augments_count", "rank_snapshot_tier",
    "rank_snapshot_division", "rank_snapshot_league_points", "rank_snapshot_wins",
    "rank_snapshot_losses", "rank_snapshot_queue_type", "rank_snapshot_retrieved_at",
    "rank_snapshot_status",
] + PROVENANCE_FIELDS
UNIT_FIELDS = [
    "match_hash", "participant_hash", "unit_index", "character_id", "rarity",
    "star_level", "item_count",
] + PROVENANCE_FIELDS
UNIT_ITEM_FIELDS = [
    "match_hash", "participant_hash", "unit_index", "item_index", "item_id", "item_name",
] + PROVENANCE_FIELDS
TRAIT_FIELDS = [
    "match_hash", "participant_hash", "trait_index", "trait_id", "units_counted",
    "style", "tier_current", "tier_total", "trait_active",
] + PROVENANCE_FIELDS
AUGMENT_FIELDS = [
    "match_hash", "participant_hash", "augment_index", "augment_id",
] + PROVENANCE_FIELDS


def normalise_matches(config: dict[str, Any], *, run_id: str) -> dict[str, list[dict[str, Any]]]:
    accepted_ids = [row["match_id"] for row in
                    read_csv_rows(project_path(config, "accepted_match_ids"), ["run_id", "match_id"])
                    if row.get("run_id") == run_id]
    if not accepted_ids:
        raise RuntimeError(f"No accepted matches for run {run_id}")
    raw_dir = project_path(config, "raw_matches")
    key = get_hash_key(config)
    snapshot_rows = [
        row for row in read_csv_rows(
            project_path(config, "rank_snapshots"),
            ["run_id", "participant_hash", "rank_snapshot_status"],
        ) if row.get("run_id") == run_id
    ]
    snapshots = {row["participant_hash"]: row for row in snapshot_rows}
    if len(snapshots) != len(snapshot_rows):
        raise RuntimeError(f"Duplicate rank snapshots found for run {run_id}")
    invalid_statuses = {
        str(row.get("rank_snapshot_status", "")) for row in snapshot_rows
    }.difference({"found", "unranked", "not_found", "request_failed"})
    if invalid_statuses:
        raise RuntimeError(
            f"Invalid rank snapshot statuses for run {run_id}: "
            f"{', '.join(sorted(invalid_statuses))}"
        )
    provenance = {
        "run_id": run_id,
        "dataset_version": config["project"]["dataset_version"],
        "schema_version": config["project"]["schema_version"],
    }
    tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for match_id in accepted_ids:
        payload = load_json(raw_dir / f"{safe_filename(match_id)}.json")
        info = payload["info"]
        match_hash = hash_identifier(match_id, key)
        participants = info["participants"]
        tables["matches"].append({
            "match_hash": match_hash,
            "game_datetime_ms": info.get("game_datetime"),
            "game_datetime_utc": epoch_ms_to_iso(info.get("game_datetime")),
            "game_length_seconds": info.get("game_length"),
            "patch": config["scope"]["study_patch"],
            "queue_id": info.get("queue_id"),
            "tft_game_type": info.get("tft_game_type"),
            "tft_set_number": info.get("tft_set_number"),
            "participant_count": len(participants), **provenance,
        })
        for participant in participants:
            puuid = str(participant["puuid"])
            participant_hash = hash_identifier(puuid, key)
            snapshot = snapshots.get(participant_hash)
            if snapshot is None:
                raise RuntimeError(
                    f"Missing rank snapshot for an accepted participant in run {run_id}"
                )
            placement = safe_int(participant.get("placement"))
            units = participant.get("units", [])
            traits = participant.get("traits", [])
            augments = participant.get("augments", []) if isinstance(participant.get("augments"), list) else []
            active_traits = sum(
                int((safe_int(trait.get("tier_current"), 0) or 0) > 0 or
                    (safe_int(trait.get("style"), 0) or 0) > 0)
                for trait in traits if isinstance(trait, dict)
            )
            tables["participants"].append({
                "match_hash": match_hash, "participant_hash": participant_hash,
                "placement": placement, "top_four": bool(placement and placement <= 4),
                "placement_group": placement_group(placement), "level": participant.get("level"),
                "gold_left": participant.get("gold_left"), "last_round": participant.get("last_round"),
                "players_eliminated": participant.get("players_eliminated"),
                "total_damage_to_players": participant.get("total_damage_to_players"),
                "time_eliminated": participant.get("time_eliminated"), "units_count": len(units),
                "traits_count": len(traits), "active_traits_count": active_traits,
                "augments_count": len(augments),
                "rank_snapshot_tier": snapshot.get("rank_snapshot_tier", ""),
                "rank_snapshot_division": snapshot.get("rank_snapshot_division", ""),
                "rank_snapshot_league_points": snapshot.get("rank_snapshot_league_points", ""),
                "rank_snapshot_wins": snapshot.get("rank_snapshot_wins", ""),
                "rank_snapshot_losses": snapshot.get("rank_snapshot_losses", ""),
                "rank_snapshot_queue_type": snapshot.get("rank_snapshot_queue_type", ""),
                "rank_snapshot_retrieved_at": snapshot.get("rank_snapshot_retrieved_at", ""),
                "rank_snapshot_status": snapshot.get("rank_snapshot_status", ""),
                **provenance,
            })
            for unit_index, unit in enumerate(units, 1):
                if not isinstance(unit, dict):
                    continue
                item_ids = unit.get("items", []) if isinstance(unit.get("items"), list) else []
                item_names = unit.get("itemNames", []) if isinstance(unit.get("itemNames"), list) else []
                item_count = max(len(item_ids), len(item_names))
                tables["units"].append({
                    "match_hash": match_hash, "participant_hash": participant_hash,
                    "unit_index": unit_index, "character_id": unit.get("character_id"),
                    "rarity": unit.get("rarity"), "star_level": unit.get("tier"),
                    "item_count": item_count, **provenance,
                })
                for item_index in range(item_count):
                    tables["unit_items"].append({
                        "match_hash": match_hash, "participant_hash": participant_hash,
                        "unit_index": unit_index, "item_index": item_index + 1,
                        "item_id": item_ids[item_index] if item_index < len(item_ids) else None,
                        "item_name": item_names[item_index] if item_index < len(item_names) else None,
                        **provenance,
                    })
            for trait_index, trait in enumerate(traits, 1):
                if not isinstance(trait, dict):
                    continue
                tier_current = safe_int(trait.get("tier_current"), 0) or 0
                style = safe_int(trait.get("style"), 0) or 0
                tables["traits"].append({
                    "match_hash": match_hash, "participant_hash": participant_hash,
                    "trait_index": trait_index, "trait_id": trait.get("name"),
                    "units_counted": trait.get("num_units"), "style": style,
                    "tier_current": tier_current, "tier_total": trait.get("tier_total"),
                    "trait_active": bool(tier_current > 0 or style > 0), **provenance,
                })
            for augment_index, augment in enumerate(augments, 1):
                tables["augments"].append({
                    "match_hash": match_hash, "participant_hash": participant_hash,
                    "augment_index": augment_index, "augment_id": augment, **provenance,
                })
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pseudonymous relational tables")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    tables = normalise_matches(config, run_id=args.run_id)
    schemas = {
        "matches": MATCH_FIELDS, "participants": PARTICIPANT_FIELDS, "units": UNIT_FIELDS,
        "unit_items": UNIT_ITEM_FIELDS, "traits": TRAIT_FIELDS, "augments": AUGMENT_FIELDS,
    }
    output_dir = project_path(config, "interim") / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, fields in schemas.items():
        path = output_dir / f"{name}.csv"
        write_csv_rows(path, tables.get(name, []), fields)
        print(f"{name}: {len(tables.get(name, []))} rows")
    table_manifest = {
        name: {"rows": len(tables.get(name, [])), "sha256": sha256_file(output_dir / f"{name}.csv")}
        for name in schemas
    }
    statuses = [row for row in read_csv_rows(project_path(config, "collection_status"))
                if row.get("run_id") == args.run_id and row.get("raw_sha256")]
    rank_snapshot_rows = [
        row for row in read_csv_rows(project_path(config, "rank_snapshots"))
        if row.get("run_id") == args.run_id
    ]
    rank_status_counts = Counter(
        row.get("rank_snapshot_status", "") for row in rank_snapshot_rows
    )
    atomic_json_dump({
        "run_id": args.run_id,
        "dataset_version": config["project"]["dataset_version"],
        "schema_version": config["project"]["schema_version"],
        "config_hash": config_hash(config),
        "accepted_match_count": len(tables["matches"]),
        "raw_response_count": len(statuses),
        "raw_checksum_set_sha256": hashlib.sha256(
            "\n".join(sorted(row["raw_sha256"] for row in statuses)).encode("ascii")
        ).hexdigest(),
        "rank_snapshots": {
            "rows": len(rank_snapshot_rows),
            "status_counts": dict(sorted(rank_status_counts.items())),
            "partition_sha256": stable_json_hash(sorted(
                rank_snapshot_rows, key=lambda row: row.get("participant_hash", "")
            )),
        },
        "tables": table_manifest,
    }, project_path(config, "manifests") / f"dataset_{args.run_id}.json")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from typing import Any

from .api_client import (
    RiotAPIError, RiotApiClient, RiotAuthenticationError, RiotNotFoundError,
)
from .config import ensure_project_directories, load_config, project_path
from .utils import (
    epoch_ms_to_iso, get_hash_key, hash_identifier, load_json, read_csv_rows,
    replace_csv_partition, safe_filename, utc_now_iso, write_csv_rows,
)

RANK_SNAPSHOT_FIELDS = [
    "run_id", "participant_hash", "platform_region", "regional_route",
    "rank_snapshot_tier", "rank_snapshot_division",
    "rank_snapshot_league_points", "rank_snapshot_wins", "rank_snapshot_losses",
    "rank_snapshot_queue_type", "rank_snapshot_retrieved_at", "rank_snapshot_status",
    "rank_snapshot_error_class",
]
RANK_SNAPSHOT_STATUSES = {"found", "unranked", "not_found", "request_failed"}
RANK_ROUTING_VERSION = "rank_snapshot_routing_v2"
RANK_ROUTING_FIELDS = [
    "run_id", "participant_hash", "selected_platform_region",
    "selected_regional_route", "selected_from_match_id",
    "selected_match_timestamp_ms", "selected_match_timestamp_utc",
    "observed_platforms", "observed_platform_count",
    "accepted_match_occurrence_count", "platform_transfer_detected",
    "routing_version",
]


def resolve_rank_routing(
    config: dict[str, Any], run_id: str,
) -> tuple[OrderedDict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Select one deterministic lookup platform per accepted participant.

    Riot PUUIDs can legitimately span platforms after an account transfer. The
    most recent accepted match is authoritative; equal timestamps are resolved
    by lexicographically greatest immutable match_id. Earlier platforms remain
    explicit routing provenance and never create additional snapshot rows.
    """
    accepted_rows = [
        row for row in read_csv_rows(
            project_path(config, "accepted_match_ids"), ["run_id", "match_id"]
        ) if row.get("run_id") == run_id
    ]
    if not accepted_rows:
        raise RuntimeError(f"No accepted matches for run {run_id}")
    raw_dir = project_path(config, "raw_matches")
    key = get_hash_key(config)
    occurrences: dict[str, dict[str, Any]] = {}
    for accepted in accepted_rows:
        match_id = accepted["match_id"]
        platform = str(accepted.get("platform_region", "")).upper()
        route = str(accepted.get("regional_route", "")).upper()
        if not platform or not route:
            raise RuntimeError(f"Accepted match {match_id} lacks rank-snapshot routing provenance")
        payload = load_json(raw_dir / f"{safe_filename(match_id)}.json")
        timestamp = payload.get("info", {}).get("game_datetime")
        try:
            timestamp_ms = int(timestamp)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Accepted match {match_id} lacks an authoritative game_datetime"
            ) from exc
        for participant in payload.get("info", {}).get("participants", []):
            if not isinstance(participant, dict):
                continue
            puuid = str(participant.get("puuid", "")).strip()
            if not puuid:
                raise RuntimeError(f"Accepted match {match_id} contains a missing PUUID")
            participant_hash = hash_identifier(puuid, key)
            record = occurrences.setdefault(participant_hash, {
                "puuid": puuid, "occurrences": [],
            })
            if record["puuid"] != puuid:
                raise RuntimeError("Participant hash collision detected")
            record["occurrences"].append({
                "match_id": match_id, "timestamp_ms": timestamp_ms,
                "platform_region": platform, "regional_route": route,
            })

    identities: OrderedDict[str, dict[str, Any]] = OrderedDict()
    routing_rows: list[dict[str, Any]] = []
    for participant_hash in sorted(occurrences):
        record = occurrences[participant_hash]
        participant_occurrences = record["occurrences"]
        selected = max(
            participant_occurrences,
            key=lambda row: (row["timestamp_ms"], row["match_id"]),
        )
        platforms = sorted({row["platform_region"] for row in participant_occurrences})
        identities[participant_hash] = {
            "puuid": record["puuid"],
            "platform_region": selected["platform_region"],
            "regional_route": selected["regional_route"],
            "selected_from_match_id": selected["match_id"],
            "selected_match_timestamp_ms": selected["timestamp_ms"],
        }
        routing_rows.append({
            "run_id": run_id,
            "participant_hash": participant_hash,
            "selected_platform_region": selected["platform_region"],
            "selected_regional_route": selected["regional_route"],
            "selected_from_match_id": selected["match_id"],
            "selected_match_timestamp_ms": selected["timestamp_ms"],
            "selected_match_timestamp_utc": epoch_ms_to_iso(selected["timestamp_ms"]),
            "observed_platforms": platforms,
            "observed_platform_count": len(platforms),
            "accepted_match_occurrence_count": len(participant_occurrences),
            "platform_transfer_detected": len(platforms) > 1,
            "routing_version": RANK_ROUTING_VERSION,
        })
    return identities, routing_rows


def accepted_participants(config: dict[str, Any], run_id: str) -> OrderedDict[str, dict[str, Any]]:
    """Compatibility wrapper returning one routed identity per participant."""
    identities, _ = resolve_rank_routing(config, run_id)
    return identities


def collect_rank_snapshots(
    config: dict[str, Any], client: RiotApiClient, *, run_id: str,
    retry_failed: bool = False,
) -> list[dict[str, Any]]:
    identities, routing_rows = resolve_rank_routing(config, run_id)
    replace_csv_partition(
        project_path(config, "rank_snapshot_routing"), routing_rows,
        RANK_ROUTING_FIELDS, partition_field="run_id", partition_value=run_id,
    )
    path = project_path(config, "rank_snapshots")
    existing = read_csv_rows(path)
    rows_by_key: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict(
        ((row.get("run_id", ""), row.get("participant_hash", "")), row)
        for row in existing if row.get("run_id") and row.get("participant_hash")
    )
    queue_type = str(config["sampling"].get("queue_type", "RANKED_TFT")).upper()
    counts: Counter[str] = Counter()

    for index, (participant_hash, identity) in enumerate(identities.items(), start=1):
        puuid = identity["puuid"]
        platform_route = identity["platform_region"]
        regional_route = identity["regional_route"]
        key = (run_id, participant_hash)
        previous = rows_by_key.get(key)
        previous_route_matches = bool(
            previous and
            previous.get("platform_region", "").upper() == platform_route and
            previous.get("regional_route", "").upper() == regional_route
        )
        if previous and previous_route_matches and not (
            retry_failed and previous.get("rank_snapshot_status") == "request_failed"
        ):
            counts[str(previous["rank_snapshot_status"])] += 1
            continue
        retrieved_at = utc_now_iso()
        try:
            entries = client.get_league_entries_by_puuid(platform_route, puuid)
            ranked = next(
                (entry for entry in entries
                 if str(entry.get("queueType", "")).upper() == queue_type),
                None,
            )
            if ranked is None:
                row = _snapshot_row(
                    run_id, participant_hash, retrieved_at, "unranked",
                    queue_type=queue_type, platform_region=platform_route,
                    regional_route=regional_route,
                )
            else:
                row = _snapshot_row(
                    run_id, participant_hash, retrieved_at, "found", ranked=ranked,
                    queue_type=queue_type, platform_region=platform_route,
                    regional_route=regional_route,
                )
        except RiotAuthenticationError:
            raise
        except RiotNotFoundError:
            row = _snapshot_row(
                run_id, participant_hash, retrieved_at, "not_found",
                queue_type=queue_type, platform_region=platform_route,
                regional_route=regional_route,
            )
        except RiotAPIError as exc:
            row = _snapshot_row(
                run_id, participant_hash, retrieved_at, "request_failed",
                queue_type=queue_type, error_class=type(exc).__name__,
                platform_region=platform_route, regional_route=regional_route,
            )
        rows_by_key[key] = row
        counts[row["rank_snapshot_status"]] += 1
        if index % 10 == 0:
            write_csv_rows(path, rows_by_key.values(), RANK_SNAPSHOT_FIELDS)
        print(f"{index}/{len(identities)} rank snapshots {dict(sorted(counts.items()))}")

    write_csv_rows(path, rows_by_key.values(), RANK_SNAPSHOT_FIELDS)
    run_rows = [
        row for (row_run_id, _), row in rows_by_key.items() if row_run_id == run_id
    ]
    if len(run_rows) != len(identities):
        raise RuntimeError(
            f"Rank snapshot coverage mismatch: {len(run_rows)}/{len(identities)} participants"
        )
    invalid = {
        str(row.get("rank_snapshot_status", "")) for row in run_rows
    }.difference(RANK_SNAPSHOT_STATUSES)
    if invalid:
        raise RuntimeError(f"Invalid rank snapshot statuses: {', '.join(sorted(invalid))}")
    print(f"Saved {len(run_rows)} rank snapshots: {dict(sorted(counts.items()))}")
    return run_rows


def _snapshot_row(
    run_id: str, participant_hash: str, retrieved_at: str, status: str, *,
    ranked: dict[str, Any] | None = None, queue_type: str = "", error_class: str = "",
    platform_region: str = "", regional_route: str = "",
) -> dict[str, Any]:
    ranked = ranked or {}
    return {
        "run_id": run_id,
        "participant_hash": participant_hash,
        "platform_region": platform_region,
        "regional_route": regional_route,
        "rank_snapshot_tier": ranked.get("tier", ""),
        "rank_snapshot_division": ranked.get("rank", ""),
        "rank_snapshot_league_points": ranked.get("leaguePoints", ""),
        "rank_snapshot_wins": ranked.get("wins", ""),
        "rank_snapshot_losses": ranked.get("losses", ""),
        "rank_snapshot_queue_type": ranked.get("queueType", queue_type),
        "rank_snapshot_retrieved_at": retrieved_at,
        "rank_snapshot_status": status,
        "rank_snapshot_error_class": error_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect one explicit ranked-status snapshot per accepted participant"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Explicitly retry only rows previously recorded as request_failed",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    with RiotApiClient(config) as client:
        collect_rank_snapshots(
            config, client, run_id=args.run_id, retry_failed=args.retry_failed
        )


if __name__ == "__main__":
    main()

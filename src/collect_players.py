from __future__ import annotations

import argparse
import random
from collections import OrderedDict
from typing import Any

from .api_client import RiotAPIError, RiotApiClient, RiotNotFoundError
from .config import ensure_project_directories, load_config, project_path, route_for_platform
from .utils import get_hash_key, hash_identifier, read_csv_rows, replace_csv_partition, utc_now_iso

SEED_FIELDS = [
    "run_id", "participant_hash", "puuid", "game_name", "tag_line", "platform_route",
    "platform_region", "regional_route",
    "sampling_tier", "sampling_division", "sampling_page", "selection_probability",
    "league_points", "wins", "losses", "queue_type", "rank_retrieval_datetime", "source",
]
PLAYER_SAMPLING_FIELDS = [
    "run_id", "mode", "report_scope", "platform_region", "regional_route",
    "ladder_entries_returned", "valid_player_entries",
    "duplicate_player_entries_removed", "unique_seed_players_available",
    "unique_players_selected", "pages_requested", "pages_successfully_returned",
    "pages_with_entries", "pages_exhausted",
]
PLAYER_PAGE_FIELDS = [
    "run_id", "platform_region", "regional_route", "tier", "division", "page",
    "request_status", "entries_returned",
    "valid_player_entries", "new_unique_players_available", "duplicate_player_entries_removed",
    "players_selected", "pagination_exhausted",
]
APEX_TIERS = {"MASTER", "GRANDMASTER", "CHALLENGER"}


def select_ranked_entry(entries: list[dict[str, Any]], queue_type: str) -> dict[str, Any] | None:
    return next((entry for entry in entries
                 if str(entry.get("queueType", "")).upper() == queue_type.upper()), None)


def collect_manual_players(
    config: dict[str, Any], client: RiotApiClient, *, run_id: str
) -> list[dict[str, Any]]:
    rows = read_csv_rows(project_path(config, "input_seed_players"),
                         ["game_name", "tag_line", "platform_route", "puuid"])
    if not rows:
        raise RuntimeError("No manual seed players found")
    default_platform = _platform_specs(config)[0]["platform_region"]
    queue_type = config["sampling"].get("queue_type", "RANKED_TFT")
    key = get_hash_key(config)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        game_name, tag_line = row.get("game_name", "").strip(), row.get("tag_line", "").strip()
        puuid = row.get("puuid", "").strip()
        platform = row.get("platform_route", "").strip() or default_platform
        regional_route = route_for_platform(config, platform)
        try:
            if not puuid:
                if not game_name or not tag_line:
                    raise ValueError(f"row {index} needs a Riot ID or PUUID")
                puuid = str(client.get_account_by_riot_id(regional_route, game_name, tag_line).get("puuid", "")).strip()
            if not puuid or puuid in seen:
                continue
            ranked = select_ranked_entry(client.get_league_entries_by_puuid(platform, puuid), queue_type) or {}
            seen.add(puuid)
            output.append(_seed_row(run_id, puuid, key, platform, ranked,
                                    row.get("sampling_tier", ""), row.get("sampling_division", ""),
                                    "", 1.0, game_name, tag_line,
                                    "manual_riot_id" if game_name else "manual_puuid", queue_type,
                                    regional_route=regional_route))
        except (RiotNotFoundError, ValueError) as exc:
            print(f"Skipping manual row {index}: {exc}")
        except RiotAPIError as exc:
            raise RuntimeError(f"Manual seed collection aborted at row {index}: {exc}") from exc
    return output


def collect_ladder_players(
    config: dict[str, Any], client: RiotApiClient, *, run_id: str,
    stats: dict[str, Any] | None = None, page_stats: list[dict[str, Any]] | None = None,
    regional_stats: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    queue_type = config["sampling"].get("queue_type", "RANKED_TFT")
    key = get_hash_key(config)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    available_players: set[str] = set()
    ladder_entries_returned = 0
    valid_player_entries = 0
    pages_requested = 0
    pages_with_entries = 0
    pages_exhausted = 0

    region_rows: list[dict[str, Any]] = []
    platform_specs = _platform_specs(config)
    # Interleave regions within each rank stratum so discovery is balanced by
    # design rather than exhausting one ladder before touching the next.
    for specification in config["sampling"].get("rank_strata", []):
      for platform_spec in platform_specs:
        platform = platform_spec["platform_region"]
        regional_route = platform_spec["regional_route"]
        per_stratum = int(platform_spec["seed_players_per_page"])
        rng = random.Random(f"{config['sampling']['random_seed']}:{platform}")
        tier = str(specification.get("tier", "")).upper()
        divisions = [str(value).upper() for value in specification.get("divisions", [])]
        pages = (int(platform_spec["apex_pages"]) if tier in APEX_TIERS
                 else int(platform_spec["ladder_pages_per_division"]))
        strata = [("", page) for page in range(1, pages + 1)] if tier in APEX_TIERS else [
            (division, page) for division in divisions for page in range(1, pages + 1)]
        region = next((row for row in region_rows if row["platform_region"] == platform), None)
        if region is None:
            region = _empty_player_stats(run_id, platform, regional_route)
            region_rows.append(region)
        for division, page in strata:
            pages_requested += 1
            region["pages_requested"] += 1
            try:
                if tier in APEX_TIERS:
                    candidates = [dict(entry) for entry in
                                  client.get_apex_league(platform, tier, queue=queue_type).get("entries", [])
                                  if isinstance(entry, dict)]
                else:
                    candidates = client.get_league_entries_by_division(
                        platform, tier, division, page=page, queue=queue_type
                    )
            except RiotAPIError as exc:
                raise RuntimeError(f"Ladder stratum {tier} {division} page {page} failed: {exc}") from exc
            ladder_entries_returned += len(candidates)
            region["ladder_entries_returned"] += len(candidates)
            pages_with_entries += int(bool(candidates))
            pages_exhausted += int(not candidates)
            region["pages_successfully_returned"] += 1
            region["pages_with_entries"] += int(bool(candidates))
            region["pages_exhausted"] += int(not candidates)
            page_ids = [str(candidate.get("puuid", "")).strip() for candidate in candidates]
            page_ids = [puuid for puuid in page_ids if puuid]
            valid_player_entries += len(page_ids)
            region["valid_player_entries"] += len(page_ids)
            new_available = set(page_ids).difference(available_players)
            page_duplicate_count = len(page_ids) - len(new_available)
            available_players.update(page_ids)
            region.setdefault("_available", set()).update(page_ids)
            region["duplicate_player_entries_removed"] += page_duplicate_count
            unique: OrderedDict[str, dict[str, Any]] = OrderedDict()
            for candidate in candidates:
                puuid = str(candidate.get("puuid", "")).strip()
                if puuid and puuid not in seen:
                    unique.setdefault(puuid, candidate)
            pool = list(unique.values())
            rng.shuffle(pool)
            chosen = pool[:per_stratum]
            probability = min(1.0, per_stratum / len(pool)) if pool else 0.0
            if len(chosen) < per_stratum:
                print(f"Warning: {tier} {division or 'APEX'} page {page} supplied {len(chosen)}/{per_stratum} seeds")
            for entry in chosen:
                puuid = str(entry["puuid"])
                seen.add(puuid)
                selected.append(_seed_row(run_id, puuid, key, platform, entry, tier, division,
                                          page, probability, "", "", "ranked_ladder", queue_type,
                                          regional_route=regional_route))
                region["unique_players_selected"] += 1
            if page_stats is not None:
                page_stats.append({
                    "run_id": run_id, "platform_region": platform,
                    "regional_route": regional_route, "tier": tier, "division": division,
                    "page": page, "request_status": "success",
                    "entries_returned": len(candidates),
                    "valid_player_entries": len(page_ids),
                    "new_unique_players_available": len(new_available),
                    "duplicate_player_entries_removed": page_duplicate_count,
                    "players_selected": len(chosen),
                    "pagination_exhausted": not candidates,
                })
    if stats is not None:
        stats.update({
            "run_id": run_id, "mode": "ladder", "report_scope": "global",
            "platform_region": "ALL", "regional_route": "MULTI",
            "ladder_entries_returned": ladder_entries_returned,
            "valid_player_entries": valid_player_entries,
            "duplicate_player_entries_removed": valid_player_entries - len(available_players),
            "unique_seed_players_available": len(available_players),
            "unique_players_selected": len(selected),
            "pages_requested": pages_requested,
            "pages_successfully_returned": pages_requested,
            "pages_with_entries": pages_with_entries,
            "pages_exhausted": pages_exhausted,
        })
    for row in region_rows:
        row["unique_seed_players_available"] = len(row.pop("_available", set()))
    if regional_stats is not None:
        regional_stats.extend(region_rows)
    return selected


def _platform_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    enabled = [row for row in config.get("platforms", [])
               if isinstance(row, dict) and row.get("enabled")]
    if enabled:
        return [{**row, "platform_region": str(row["platform_region"]).upper(),
                 "regional_route": str(row["regional_route"]).upper()} for row in enabled]
    sampling, scope = config["sampling"], config["scope"]
    return [{"platform_region": str(scope["platform_route"]).upper(),
             "regional_route": str(scope.get("regional_route", "EUROPE")).upper(),
             "seed_players_per_page": int(sampling["seed_players_per_stratum"]),
             "ladder_pages_per_division": max(row.get("pages_per_division", 1)
                                                for row in sampling["rank_strata"]),
             "apex_pages": 1}]


def _empty_player_stats(run_id: str, platform: str, route: str) -> dict[str, Any]:
    return {"run_id": run_id, "mode": "ladder", "report_scope": "region",
            "platform_region": platform, "regional_route": route,
            "ladder_entries_returned": 0, "valid_player_entries": 0,
            "duplicate_player_entries_removed": 0, "unique_seed_players_available": 0,
            "unique_players_selected": 0, "pages_requested": 0,
            "pages_successfully_returned": 0, "pages_with_entries": 0,
            "pages_exhausted": 0}


def _seed_row(
    run_id: str, puuid: str, key: bytes, platform: str, ranked: dict[str, Any],
    tier: Any, division: Any, page: Any, probability: float, game_name: str,
    tag_line: str, source: str, queue_type: str, *, regional_route: str = "",
) -> dict[str, Any]:
    return {
        "run_id": run_id, "participant_hash": hash_identifier(puuid, key), "puuid": puuid,
        "game_name": game_name, "tag_line": tag_line, "platform_route": platform.upper(),
        "platform_region": platform.upper(), "regional_route": regional_route.upper(),
        "sampling_tier": ranked.get("tier") or tier,
        "sampling_division": ranked.get("rank") or division,
        "sampling_page": page, "selection_probability": round(probability, 8),
        "league_points": ranked.get("leaguePoints", ""), "wins": ranked.get("wins", ""),
        "losses": ranked.get("losses", ""), "queue_type": ranked.get("queueType", queue_type),
        "rank_retrieval_datetime": utc_now_iso(), "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect reproducible TFT seed players")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["ladder", "manual"], default="ladder")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    stats: dict[str, Any] = {}
    regional_stats: list[dict[str, Any]] = []
    page_stats: list[dict[str, Any]] = []
    with RiotApiClient(config) as client:
        rows = (collect_ladder_players(
                    config, client, run_id=args.run_id, stats=stats, page_stats=page_stats,
                    regional_stats=regional_stats)
                if args.mode == "ladder" else
                collect_manual_players(config, client, run_id=args.run_id))
    if args.mode == "manual":
        stats = {
            "run_id": args.run_id, "mode": "manual", "ladder_entries_returned": 0,
            "valid_player_entries": len(rows), "duplicate_player_entries_removed": 0,
            "unique_seed_players_available": len(rows), "unique_players_selected": len(rows),
            "pages_requested": 0, "pages_successfully_returned": 0,
            "pages_with_entries": 0, "pages_exhausted": 0,
        }
    replace_csv_partition(project_path(config, "seed_players"), rows, SEED_FIELDS,
                          partition_field="run_id", partition_value=args.run_id)
    replace_csv_partition(project_path(config, "player_sampling_report"), [stats, *regional_stats],
                          PLAYER_SAMPLING_FIELDS,
                          partition_field="run_id", partition_value=args.run_id)
    replace_csv_partition(project_path(config, "player_sampling_pages"), page_stats,
                          PLAYER_PAGE_FIELDS,
                          partition_field="run_id", partition_value=args.run_id)
    print(f"Saved {len(rows)} seed players")


if __name__ == "__main__":
    main()

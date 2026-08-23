from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from typing import Any

from .api_client import RiotAPIError, RiotApiClient, RiotNotFoundError
from .config import (
    SUPPORTED_PLATFORM_ROUTES, enabled_platforms, ensure_project_directories,
    load_config, project_path,
)
from .utils import read_csv_rows, replace_csv_partition, utc_now_iso

SOURCE_FIELDS = [
    "run_id", "match_id", "source_seed_hash", "source_sampling_tier",
    "source_sampling_division", "source_platform_region", "source_regional_route",
    "discovery_source", "history_page", "history_offset", "retrieval_datetime",
]
UNIQUE_FIELDS = [
    "run_id", "match_id", "source_seed_count", "source_tiers",
    "source_divisions", "platform_region", "regional_route", "source_platform_regions",
    "source_regional_routes", "discovery_paths", "first_retrieved_at",
]
CANDIDATE_REPORT_FIELDS = [
    "run_id", "report_scope", "platform_region", "regional_route", "seed_players",
    "history_pages_requested_per_seed",
    "candidate_occurrences", "unique_seed_match_associations",
    "unique_candidate_match_ids", "duplicate_candidate_occurrences_removed",
    "duplicate_candidate_rate_percentage", "cross_region_overlap_candidates",
    "out_of_scope_platform_candidate_ids",
    "unique_candidate_threshold",
    "capacity_gate_result",
]


def collect_match_ids(
    config: dict[str, Any], client: RiotApiClient, *, run_id: str,
    page_size: int, pages_per_seed: int, start_time: int | None, end_time: int | None,
    stats: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seeds = [row for row in read_csv_rows(project_path(config, "seed_players"),
                                           ["run_id", "puuid", "participant_hash"])
             if row.get("run_id") == run_id]
    if not seeds:
        raise RuntimeError(f"No seed players found for run {run_id}")
    source_map: OrderedDict[tuple[str, str, str], dict[str, Any]] = OrderedDict()
    failures = 0
    candidate_occurrences = 0
    region_occurrences: Counter[str] = Counter()
    for index, seed in enumerate(seeds, start=1):
        puuid, seed_hash = seed["puuid"].strip(), seed["participant_hash"].strip()
        platform_region = (seed.get("platform_region") or seed.get("platform_route") or
                           config.get("scope", {}).get("platform_route", "")).upper()
        regional_route = (seed.get("regional_route") or
                          config.get("scope", {}).get("regional_route", "")).upper()
        if not platform_region or not regional_route:
            raise RuntimeError(f"Seed {seed_hash} is missing explicit region routing")
        try:
            for page in range(pages_per_seed):
                offset = page * page_size
                ids = client.get_match_ids(regional_route, puuid, start=offset, count=page_size,
                                           start_time=start_time, end_time=end_time)
                retrieved_at = utc_now_iso()
                candidate_occurrences += len(ids)
                region_occurrences[platform_region] += len(ids)
                for match_id in ids:
                    source_map.setdefault((match_id, seed_hash, platform_region), {
                        "run_id": run_id, "match_id": match_id,
                        "source_seed_hash": seed_hash,
                        "source_sampling_tier": seed.get("sampling_tier", ""),
                        "source_sampling_division": seed.get("sampling_division", ""),
                        "source_platform_region": platform_region,
                        "source_regional_route": regional_route,
                        "discovery_source": seed.get("source", "ranked_ladder"),
                        "history_page": page + 1, "history_offset": offset,
                        "retrieval_datetime": retrieved_at,
                    })
                if len(ids) < page_size:
                    break
            print(f"Seed {index}/{len(seeds)} complete")
        except RiotNotFoundError:
            print(f"Seed {index}/{len(seeds)} has no match history")
        except RiotAPIError as exc:
            failures += 1
            print(f"Seed {index}/{len(seeds)} failed: {exc}")
    if failures:
        raise RuntimeError(f"Match-ID discovery incomplete: {failures}/{len(seeds)} seeds failed")

    source_rows = list(source_map.values())
    aggregate: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in source_rows:
        record = aggregate.setdefault(row["match_id"], {
            "seed_hashes": set(), "tiers": set(), "divisions": set(),
            "platforms": set(), "routes": set(), "paths": set(),
            "platform_routes": {},
            "first_retrieved_at": row["retrieval_datetime"],
        })
        record["seed_hashes"].add(row["source_seed_hash"])
        if row["source_sampling_tier"]:
            record["tiers"].add(row["source_sampling_tier"])
        if row["source_sampling_division"]:
            record["divisions"].add(row["source_sampling_division"])
        record["platforms"].add(row["source_platform_region"])
        record["routes"].add(row["source_regional_route"])
        record["platform_routes"][row["source_platform_region"]] = row["source_regional_route"]
        record["paths"].add(
            f"{row['source_platform_region']}:{row['source_sampling_tier']}:"
            f"{row['source_sampling_division'] or 'APEX'}:{row['source_seed_hash']}"
        )
        record["first_retrieved_at"] = min(record["first_retrieved_at"], row["retrieval_datetime"])
    unique_rows = []
    out_of_scope_ids: set[str] = set()
    for match_id, record in aggregate.items():
        routing = _candidate_route(config, match_id, record)
        if routing is None:
            out_of_scope_ids.add(match_id)
            continue
        platform, route = routing
        unique_rows.append({
            "run_id": run_id, "match_id": match_id,
            "source_seed_count": len(record["seed_hashes"]),
            "source_tiers": sorted(record["tiers"]),
            "source_divisions": sorted(record["divisions"]),
            "platform_region": platform,
            "regional_route": route,
            "source_platform_regions": sorted(record["platforms"]),
            "source_regional_routes": sorted(record["routes"]),
            "discovery_paths": sorted(record["paths"]),
            "first_retrieved_at": record["first_retrieved_at"],
        })
    if stats is not None:
        threshold = int(config.get("sampling", {}).get("candidate_pool_capacity_target", 0))
        duplicates_removed = candidate_occurrences - len(aggregate)
        stats.update({
            "run_id": run_id, "report_scope": "global", "platform_region": "ALL",
            "regional_route": "MULTI",
            "seed_players": len(seeds),
            "history_pages_requested_per_seed": pages_per_seed,
            "candidate_occurrences": candidate_occurrences,
            "unique_seed_match_associations": len(source_rows),
            "unique_candidate_match_ids": len(unique_rows),
            "duplicate_candidate_occurrences_removed": duplicates_removed,
            "duplicate_candidate_rate_percentage": round(
                100 * duplicates_removed / candidate_occurrences, 4
            ) if candidate_occurrences else 0.0,
            "cross_region_overlap_candidates": sum(
                match_id not in out_of_scope_ids and len(record["platforms"]) > 1
                for match_id, record in aggregate.items()
            ),
            "out_of_scope_platform_candidate_ids": len(out_of_scope_ids),
            "unique_candidate_threshold": threshold,
            "capacity_gate_result": "pass" if len(unique_rows) >= threshold else "fail",
        })
        regional_rows = _regional_candidate_stats(
            run_id, seeds, source_rows, aggregate, region_occurrences, pages_per_seed,
            threshold, out_of_scope_ids,
        )
        stats["regional_rows"] = regional_rows
    return source_rows, unique_rows


def _candidate_route(config: dict[str, Any], match_id: str,
                     record: dict[str, Any]) -> tuple[str, str] | None:
    """Resolve download routing from the match ID, not the discovering seed.

    Match-history endpoints are regional-cluster scoped. A seed sampled on ME1
    can therefore return an EUW1-prefixed match (for example after a platform
    transfer). Riot's match ID prefix is the authoritative platform provenance;
    the seed platform remains separately recorded in source_platform_regions.
    """
    prefix = match_id.split("_", 1)[0].upper()
    if prefix in SUPPORTED_PLATFORM_ROUTES:
        if config.get("platforms"):
            configured = {row["platform_region"]: row["regional_route"]
                          for row in enabled_platforms(config)}
            if prefix not in configured:
                # Retain its discovery-source row for audit, but do not permit a
                # migrated player's out-of-scope platform match into the pool.
                return None
            return prefix, configured[prefix]
        return prefix, SUPPORTED_PLATFORM_ROUTES[prefix]

    # Legacy/synthetic IDs have no Riot platform prefix. Preserve the previous
    # source-based fallback for offline tests and backwards-compatible tooling.
    platform = sorted(record["platforms"])[0]
    return platform, record["platform_routes"][platform]


def _regional_candidate_stats(run_id: str, seeds: list[dict[str, str]],
                              sources: list[dict[str, Any]], aggregate: OrderedDict,
                              occurrences: Counter[str], pages: int, threshold: int,
                              out_of_scope_ids: set[str]) -> list[dict[str, Any]]:
    rows = []
    for platform in sorted({seed.get("platform_region") or seed.get("platform_route", "") for seed in seeds}):
        region_sources = [row for row in sources if row["source_platform_region"] == platform]
        all_ids = {row["match_id"] for row in region_sources}
        ids = all_ids.difference(out_of_scope_ids)
        count = occurrences[platform]
        route = next((row["source_regional_route"] for row in region_sources), "")
        overlap = sum(len(aggregate[match_id]["platforms"]) > 1 for match_id in ids)
        rows.append({"run_id": run_id, "report_scope": "region",
                     "platform_region": platform, "regional_route": route,
                     "seed_players": sum((seed.get("platform_region") or seed.get("platform_route")) == platform
                                         for seed in seeds),
                     "history_pages_requested_per_seed": pages,
                     "candidate_occurrences": count,
                     "unique_seed_match_associations": len(region_sources),
                     "unique_candidate_match_ids": len(ids),
                     "duplicate_candidate_occurrences_removed": count - len(all_ids),
                     "duplicate_candidate_rate_percentage": round(
                         100 * (count-len(all_ids))/count, 4
                     ) if count else 0.0,
                     "cross_region_overlap_candidates": overlap,
                     "out_of_scope_platform_candidate_ids": len(
                         all_ids.intersection(out_of_scope_ids)
                     ),
                     "unique_candidate_threshold": threshold,
                     "capacity_gate_result": "informational"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover paginated TFT match IDs")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--page-size", type=int, default=None)
    parser.add_argument("--pages-per-seed", type=int, default=None)
    parser.add_argument("--start-time", type=int, default=None)
    parser.add_argument("--end-time", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    sampling, scope = config["sampling"], config["scope"]
    page_size = args.page_size or int(sampling["match_history_page_size"])
    pages = args.pages_per_seed or int(sampling["match_history_pages_per_seed"])
    start_time = args.start_time if args.start_time is not None else scope.get("start_time")
    end_time = args.end_time if args.end_time is not None else scope.get("end_time")
    if start_time is None or end_time is None:
        raise RuntimeError("Verified scope.start_time and scope.end_time are required")
    stats: dict[str, Any] = {}
    with RiotApiClient(config) as client:
        sources, unique = collect_match_ids(
            config, client, run_id=args.run_id, page_size=page_size,
            pages_per_seed=pages, start_time=int(start_time), end_time=int(end_time),
            stats=stats,
        )
    replace_csv_partition(project_path(config, "match_id_sources"), sources, SOURCE_FIELDS,
                          partition_field="run_id", partition_value=args.run_id)
    replace_csv_partition(project_path(config, "unique_match_ids"), unique, UNIQUE_FIELDS,
                          partition_field="run_id", partition_value=args.run_id)
    regional_rows = stats.pop("regional_rows", [])
    replace_csv_partition(project_path(config, "candidate_discovery_report"), [stats, *regional_rows],
                          CANDIDATE_REPORT_FIELDS,
                          partition_field="run_id", partition_value=args.run_id)
    print(f"Saved {len(unique)} unique candidate match IDs")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import OrderedDict
from typing import Any

from .api_client import RiotAPIError, RiotApiClient, RiotNotFoundError
from .config import ensure_project_directories, load_config, project_path
from .utils import read_csv_rows, replace_csv_partition, utc_now_iso

SOURCE_FIELDS = [
    "run_id", "match_id", "source_seed_hash", "source_sampling_tier",
    "source_sampling_division", "history_page", "history_offset", "retrieval_datetime",
]
UNIQUE_FIELDS = [
    "run_id", "match_id", "source_seed_count", "source_tiers",
    "source_divisions", "first_retrieved_at",
]
CANDIDATE_REPORT_FIELDS = [
    "run_id", "seed_players", "history_pages_requested_per_seed",
    "candidate_occurrences", "unique_seed_match_associations",
    "unique_candidate_match_ids", "duplicate_candidate_occurrences_removed",
    "duplicate_candidate_rate_percentage", "unique_candidate_threshold",
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
    regional_route = config["scope"]["regional_route"]
    source_map: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    failures = 0
    candidate_occurrences = 0
    for index, seed in enumerate(seeds, start=1):
        puuid, seed_hash = seed["puuid"].strip(), seed["participant_hash"].strip()
        try:
            for page in range(pages_per_seed):
                offset = page * page_size
                ids = client.get_match_ids(regional_route, puuid, start=offset, count=page_size,
                                           start_time=start_time, end_time=end_time)
                retrieved_at = utc_now_iso()
                candidate_occurrences += len(ids)
                for match_id in ids:
                    source_map.setdefault((match_id, seed_hash), {
                        "run_id": run_id, "match_id": match_id,
                        "source_seed_hash": seed_hash,
                        "source_sampling_tier": seed.get("sampling_tier", ""),
                        "source_sampling_division": seed.get("sampling_division", ""),
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
            "first_retrieved_at": row["retrieval_datetime"],
        })
        record["seed_hashes"].add(row["source_seed_hash"])
        if row["source_sampling_tier"]:
            record["tiers"].add(row["source_sampling_tier"])
        if row["source_sampling_division"]:
            record["divisions"].add(row["source_sampling_division"])
        record["first_retrieved_at"] = min(record["first_retrieved_at"], row["retrieval_datetime"])
    unique_rows = [{
        "run_id": run_id, "match_id": match_id,
        "source_seed_count": len(record["seed_hashes"]),
        "source_tiers": sorted(record["tiers"]),
        "source_divisions": sorted(record["divisions"]),
        "first_retrieved_at": record["first_retrieved_at"],
    } for match_id, record in aggregate.items()]
    if stats is not None:
        threshold = int(config.get("sampling", {}).get("candidate_pool_capacity_target", 0))
        duplicates_removed = candidate_occurrences - len(unique_rows)
        stats.update({
            "run_id": run_id,
            "seed_players": len(seeds),
            "history_pages_requested_per_seed": pages_per_seed,
            "candidate_occurrences": candidate_occurrences,
            "unique_seed_match_associations": len(source_rows),
            "unique_candidate_match_ids": len(unique_rows),
            "duplicate_candidate_occurrences_removed": duplicates_removed,
            "duplicate_candidate_rate_percentage": round(
                100 * duplicates_removed / candidate_occurrences, 4
            ) if candidate_occurrences else 0.0,
            "unique_candidate_threshold": threshold,
            "capacity_gate_result": "pass" if len(unique_rows) >= threshold else "fail",
        })
    return source_rows, unique_rows


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
    replace_csv_partition(project_path(config, "candidate_discovery_report"), [stats],
                          CANDIDATE_REPORT_FIELDS,
                          partition_field="run_id", partition_value=args.run_id)
    print(f"Saved {len(unique)} unique candidate match IDs")


if __name__ == "__main__":
    main()

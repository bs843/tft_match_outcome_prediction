from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config import project_path
from .utils import atomic_json_dump, read_csv_rows


def write_collection_report(config: dict[str, Any], run_id: str) -> Path:
    """Write aggregate, identifier-free collection provenance for one run."""
    manifest_path = project_path(config, "manifests") / f"collection_run_{run_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    player_rows = _partition(config, "player_sampling_report", run_id)
    page_rows = _partition(config, "player_sampling_pages", run_id)
    seed_rows = _partition(config, "seed_players", run_id)
    discovery_rows = _partition(config, "candidate_discovery_report", run_id)
    status_rows = _partition(config, "collection_status", run_id)
    rank_rows = _partition(config, "rank_snapshots", run_id)
    rejection_counts: Counter[str] = Counter()
    for row in status_rows:
        if str(row.get("validation_accepted", "")).lower() == "true":
            continue
        try:
            reasons = json.loads(row.get("validation_reasons", "") or "[]")
        except json.JSONDecodeError:
            reasons = ["unparseable_validation_reason"]
        rejection_counts.update(str(reason) for reason in reasons)
    accepted = sum(str(row.get("validation_accepted", "")).lower() == "true" for row in status_rows)
    evaluated = sum(bool(row.get("raw_file")) for row in status_rows)
    requested = int(manifest.get("requested_accepted_match_target") or 0)
    seed_counts = Counter(
        f"{row.get('sampling_tier', 'UNKNOWN')}:{row.get('sampling_division', '') or 'APEX'}"
        for row in seed_rows
    )
    participant_integrity_reasons = {
        "duplicate_participant_puuid", "metadata_participant_mismatch",
        "unexpected_participant_count", "incomplete_or_duplicate_placements",
        "invalid_participant_record",
    }
    participant_integrity_failures = sum(
        count for reason, count in rejection_counts.items() if reason in participant_integrity_reasons
    )
    interim_participants = project_path(config, "interim") / run_id / "participants.csv"
    expected_rank_ids = {
        row.get("participant_hash", "") for row in read_csv_rows(interim_participants)
        if row.get("participant_hash", "")
    }
    observed_rank_ids = {row.get("participant_hash", "") for row in rank_rows if row.get("participant_hash", "")}
    regional_diagnostics: dict[str, Any] = {}
    for platform in [row.get("platform_region", "") for row in config.get("platforms", [])
                     if row.get("enabled")]:
        regional_players = [row for row in player_rows if row.get("platform_region") == platform]
        regional_pages = [row for row in page_rows if row.get("platform_region") == platform]
        regional_seeds = [row for row in seed_rows if row.get("platform_region") == platform]
        regional_discovery = [row for row in discovery_rows if row.get("platform_region") == platform]
        regional_status = [row for row in status_rows if row.get("platform_region") == platform]
        regional_rank = [row for row in rank_rows if row.get("platform_region") == platform]
        evaluated_region = sum(bool(row.get("raw_file")) for row in regional_status)
        accepted_region = sum(str(row.get("validation_accepted", "")).lower() == "true"
                              for row in regional_status)
        region_rejections: Counter[str] = Counter()
        for row in regional_status:
            if str(row.get("validation_accepted", "")).lower() == "true":
                continue
            try:
                region_rejections.update(json.loads(row.get("validation_reasons", "") or "[]"))
            except json.JSONDecodeError:
                region_rejections["unparseable_validation_reason"] += 1
        regional_diagnostics[platform] = {
            "regional_route": next((row.get("regional_route", "") for row in regional_seeds), ""),
            "ladder_pages_requested": _integer(regional_players, "pages_requested"),
            "populated_ladder_pages": sum(str(row.get("pagination_exhausted", "")).lower() != "true"
                                           for row in regional_pages),
            "ladder_entries_returned": _integer(regional_players, "ladder_entries_returned"),
            "unique_ladder_players": _integer(regional_players, "unique_seed_players_available"),
            "unique_seed_players_selected": len(regional_seeds),
            "seed_players_by_rank_division": dict(sorted(Counter(
                f"{row.get('sampling_tier', 'UNKNOWN')}:{row.get('sampling_division', '') or 'APEX'}"
                for row in regional_seeds).items())),
            "candidate_occurrences": _integer(regional_discovery, "candidate_occurrences"),
            "unique_candidate_ids_contributed": _integer(regional_discovery, "unique_candidate_match_ids"),
            "duplicate_candidate_occurrences": _integer(regional_discovery, "duplicate_candidate_occurrences_removed"),
            "cross_region_overlap_candidates": _integer(regional_discovery, "cross_region_overlap_candidates"),
            "accepted_matches": accepted_region,
            "rejection_counts": dict(sorted(region_rejections.items())),
            "acceptance_rate_percentage": round(100 * accepted_region / evaluated_region, 4)
                                          if evaluated_region else 0.0,
            "rank_snapshot_status_counts": dict(sorted(Counter(
                row.get("rank_snapshot_status", "unknown") for row in regional_rank).items())),
            "rank_snapshot_counts_by_rank_stratum": dict(sorted(Counter(
                f"{row.get('rank_snapshot_tier') or 'UNRANKED'}:"
                f"{row.get('rank_snapshot_division') or 'NONE'}" for row in regional_rank
            ).items())),
            "rank_snapshot_count": len(regional_rank),
        }
    report = {
        "run_id": run_id,
        "status": manifest.get("status", "unknown"),
        "git_commit": manifest.get("git_commit", "unavailable"),
        "git_tag": manifest.get("git_tag", "unavailable"),
        "configuration_version": config["provenance"]["configuration_version"],
        "dataset_version": manifest.get("dataset_version", config["project"]["dataset_version"]),
        "schema_version": manifest.get("schema_version", config["project"]["schema_version"]),
        "study_patch": manifest.get("study_patch", config["scope"]["study_patch"]),
        "enabled_platforms": manifest.get("enabled_platforms", []),
        "requested_accepted_matches": requested,
        "started_at": manifest.get("started_at"),
        "completed_at": manifest.get("completed_at"),
        "ladder_entries_returned": _integer(player_rows, "ladder_entries_returned"),
        "ladder_pages_requested": _integer(player_rows, "pages_requested"),
        "ladder_pages_successfully_returned": _integer(player_rows, "pages_successfully_returned"),
        "ladder_pages_with_entries": _integer(player_rows, "pages_with_entries"),
        "ladder_pages_exhausted": _integer(player_rows, "pages_exhausted"),
        "ladder_page_diagnostics_rows": len(page_rows),
        "duplicate_player_entries_removed": _integer(player_rows, "duplicate_player_entries_removed"),
        "unique_seed_players_available": _integer(player_rows, "unique_seed_players_available"),
        "unique_players_selected": _integer(player_rows, "unique_players_selected"),
        "seed_players_by_rank_division": dict(sorted(seed_counts.items())),
        "candidate_occurrences": _integer(discovery_rows, "candidate_occurrences"),
        "unique_candidate_match_ids": _integer(discovery_rows, "unique_candidate_match_ids"),
        "duplicate_candidate_occurrences_removed": _integer(
            discovery_rows, "duplicate_candidate_occurrences_removed"
        ),
        "duplicate_candidate_rate_percentage": _number(
            discovery_rows, "duplicate_candidate_rate_percentage"
        ),
        "unique_candidate_threshold": _integer(discovery_rows, "unique_candidate_threshold"),
        "capacity_gate_result": _text(discovery_rows, "capacity_gate_result"),
        "candidate_matches_evaluated": evaluated,
        "new_raw_matches_downloaded": sum(row.get("status") == "downloaded" for row in status_rows),
        "existing_verified_raw_matches_reused": sum(
            row.get("status") == "existing_verified" for row in status_rows
        ),
        "raw_checksum_failures": sum(
            row.get("error_class") == "raw_checksum_mismatch" for row in status_rows
        ),
        "accepted_matches": accepted,
        "remaining_accepted_target": max(0, requested - accepted),
        "observed_acceptance_rate_percentage": round(100 * accepted / evaluated, 4) if evaluated else 0.0,
        "accepted_patch_17_8_compliance_percentage": 100.0 if accepted else 0.0,
        "rejection_counts_by_reason": dict(sorted(rejection_counts.items())),
        "participant_integrity_failure_reasons": participant_integrity_failures,
        "rank_snapshot_rows": len(rank_rows),
        "rank_snapshot_status_counts": dict(sorted(Counter(
            row.get("rank_snapshot_status", "unknown") for row in rank_rows
        ).items())),
        "rank_snapshot_counts_by_rank_stratum": dict(sorted(Counter(
            f"{row.get('rank_snapshot_tier') or 'UNRANKED'}:"
            f"{row.get('rank_snapshot_division') or 'NONE'}" for row in rank_rows
        ).items())),
        "rank_snapshot_coverage_percentage": round(
            100 * len(expected_rank_ids.intersection(observed_rank_ids)) / len(expected_rank_ids), 4
        ) if expected_rank_ids else 0.0,
        "accepted_match_manifest_id": manifest.get("accepted_match_manifest_id"),
        "accepted_match_manifest_sha256": manifest.get("accepted_match_manifest_sha256"),
        "regional_diagnostics": regional_diagnostics,
    }
    output = project_path(config, "collection_reports") / f"collection_{run_id}.json"
    atomic_json_dump(report, output)
    return output


def _partition(config: dict[str, Any], path_key: str, run_id: str) -> list[dict[str, str]]:
    return [row for row in read_csv_rows(project_path(config, path_key)) if row.get("run_id") == run_id]


def _integer(rows: list[dict[str, str]], field: str) -> int:
    try:
        return int(rows[0].get(field, 0)) if rows else 0
    except (TypeError, ValueError):
        return 0


def _number(rows: list[dict[str, str]], field: str) -> float:
    try:
        return float(rows[0].get(field, 0)) if rows else 0.0
    except (TypeError, ValueError):
        return 0.0


def _text(rows: list[dict[str, str]], field: str) -> str:
    return str(rows[0].get(field, "")) if rows else ""

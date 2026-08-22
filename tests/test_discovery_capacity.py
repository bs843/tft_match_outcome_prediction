from __future__ import annotations

from pathlib import Path

import pytest

from src.collect_match_ids import collect_match_ids
from src.collect_matches import collect_matches
from src.collect_players import collect_ladder_players
import src.run_pipeline as pipeline
from src.run_pipeline import CandidateCapacityError, enforce_candidate_capacity, resolve_run_id
from src.utils import atomic_json_dump, read_csv_rows, sha256_file, write_csv_rows
from tests.validation.helpers import valid_match


class LadderClient:
    def __init__(self) -> None:
        self.pages: list[int] = []

    def get_league_entries_by_division(self, platform, tier, division, *, page, queue):
        self.pages.append(page)
        if page == 3:
            return []
        return [
            {"puuid": "shared-player", "tier": tier, "rank": division, "queueType": queue},
            {"puuid": f"player-{page}", "tier": tier, "rank": division, "queueType": queue},
        ]


class HistoryClient:
    def __init__(self) -> None:
        self.offsets: list[int] = []

    def get_match_ids(self, route, puuid, *, start, count, start_time, end_time):
        self.offsets.append(start)
        return {
            0: ["MATCH-A", "MATCH-B"],
            2: ["MATCH-B", "MATCH-C"],
            4: ["MATCH-C", "MATCH-D"],
        }[start]


class WideLadderClient:
    def get_league_entries_by_division(self, platform, tier, division, *, page, queue):
        return [{"puuid": f"player-{index}", "tier": tier, "rank": division,
                 "queueType": queue} for index in range(200)]


class MatchClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_match(self, route, match_id):
        self.calls.append(match_id)
        return valid_match(match_id)


def collection_config(unique: Path, status: Path, raw: Path) -> dict:
    return {
        "sampling": {"random_seed": 4275},
        "scope": {
            "regional_route": "EUROPE", "study_patch": "17.8",
            "allowed_client_patches": ["16.15"], "set_number": 17,
            "allowed_queue_ids": [1100], "allowed_game_types": ["standard"],
            "allowed_end_results": ["GameComplete"], "expected_participants": 8,
            "start_time": 1_700_000_000, "end_time": 1_800_000_000,
        },
        "validation": {
            "reject_on_wrong_set": True, "reject_on_wrong_patch": True,
            "reject_on_wrong_queue": True, "reject_on_wrong_game_type": True,
            "reject_on_wrong_end_result": True, "reject_outside_collection_window": True,
            "reject_on_unexpected_participant_count": True,
            "reject_on_invalid_participant": True, "require_complete_placement_set": True,
            "require_nonempty_units": True, "require_traits_list": True,
            "require_metadata_participant_match": True,
        },
        "paths": {
            "unique_match_ids": str(unique), "collection_status": str(status),
            "raw_matches": str(raw),
        },
    }


def test_three_ladder_pages_and_player_deduplication_are_honoured(monkeypatch) -> None:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "k" * 32)
    client = LadderClient()
    config = {
        "scope": {"platform_route": "ME1"},
        "sampling": {
            "queue_type": "RANKED_TFT", "seed_players_per_stratum": 160,
            "random_seed": 4275,
            "rank_strata": [{"tier": "GOLD", "divisions": ["I"], "pages_per_division": 3}],
        },
        "privacy": {"hash_key_env": "PARTICIPANT_HASH_KEY"},
    }
    stats = {}
    page_stats = []
    rows = collect_ladder_players(
        config, client, run_id="new-v4-run", stats=stats, page_stats=page_stats
    )
    assert client.pages == [1, 2, 3]
    assert len(rows) == 3
    assert len({row["puuid"] for row in rows}) == 3
    assert stats["ladder_entries_returned"] == 4
    assert stats["duplicate_player_entries_removed"] == 1
    assert stats["unique_seed_players_available"] == 3
    assert stats["unique_players_selected"] == 3
    assert stats["pages_exhausted"] == 1
    assert len(page_stats) == 3
    assert page_stats[-1]["pagination_exhausted"] is True


def test_seed_selection_is_capped_at_160_per_populated_page(monkeypatch) -> None:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "k" * 32)
    config = {
        "scope": {"platform_route": "ME1"},
        "sampling": {
            "queue_type": "RANKED_TFT", "seed_players_per_stratum": 160,
            "random_seed": 4275,
            "rank_strata": [{"tier": "GOLD", "divisions": ["I"], "pages_per_division": 1}],
        },
        "privacy": {"hash_key_env": "PARTICIPANT_HASH_KEY"},
    }
    rows = collect_ladder_players(config, WideLadderClient(), run_id="new-v4-run")
    assert len(rows) == 160
    assert len({row["puuid"] for row in rows}) == 160


def test_three_history_pages_deduplicate_candidates(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir()
    seeds = private / "seeds.csv"
    write_csv_rows(seeds, [{
        "run_id": "new-v3-run", "puuid": "private-player", "participant_hash": "hash",
        "sampling_tier": "GOLD", "sampling_division": "I",
        "platform_region": "ME1", "regional_route": "EUROPE", "source": "ranked_ladder",
    }], ["run_id", "puuid", "participant_hash", "sampling_tier", "sampling_division",
          "platform_region", "regional_route", "source"])
    config = {"paths": {"seed_players": str(seeds)}}
    client = HistoryClient()
    stats = {}
    sources, unique = collect_match_ids(
        config, client, run_id="new-v3-run", page_size=2, pages_per_seed=3,
        start_time=1, end_time=2, stats=stats,
    )
    assert client.offsets == [0, 2, 4]
    assert len(sources) == 4
    assert [row["match_id"] for row in unique] == ["MATCH-A", "MATCH-B", "MATCH-C", "MATCH-D"]
    assert stats["candidate_occurrences"] == 6
    assert stats["unique_candidate_match_ids"] == 4
    assert stats["duplicate_candidate_occurrences_removed"] == 2


@pytest.mark.parametrize("failed_run", [
    "20260820T152821Z-9df0ecbb",
    "20260820T212450Z-31431b86",
])
def test_failed_prior_run_ids_cannot_be_resumed_or_merged(failed_run: str) -> None:
    with pytest.raises(RuntimeError, match="cannot be resumed or merged"):
        resolve_run_id("main", failed_run)


def test_25000_candidate_capacity_gate_passes_and_fails(tmp_path: Path) -> None:
    unique = tmp_path / "unique.csv"
    write_csv_rows(unique, [
        {"run_id": "run", "match_id": "A"},
        {"run_id": "run", "match_id": "B"},
    ], ["run_id", "match_id"])
    config = {
        "sampling": {"candidate_pool_capacity_target": 25000},
        "paths": {"unique_match_ids": str(unique)},
    }
    with pytest.raises(CandidateCapacityError, match="at least 25000"):
        enforce_candidate_capacity(config, "run")
    config["sampling"]["candidate_pool_capacity_target"] = 2
    assert enforce_candidate_capacity(config, "run") == 2


def test_capacity_failure_stops_before_download_manifest_and_rank(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "run_module", lambda module, *args: calls.append(module))
    monkeypatch.setattr(
        pipeline, "enforce_candidate_capacity",
        lambda config, run_id: (_ for _ in ()).throw(CandidateCapacityError("capacity")),
    )
    with pytest.raises(CandidateCapacityError):
        pipeline.run_acquisition_stages(
            {}, ["--run-id", "new-v4-run"], run_id="new-v4-run",
            player_mode="ladder", accepted_target=5000, enforce_capacity=True,
        )
    assert calls == ["src.collect_players", "src.collect_match_ids"]
    assert "src.collect_matches" not in calls
    assert "src.validate_matches" not in calls
    assert "src.collect_rank_snapshots" not in calls


def test_duplicate_candidate_rows_are_downloaded_and_counted_once(tmp_path: Path) -> None:
    unique = tmp_path / "unique.csv"
    status = tmp_path / "status.csv"
    raw = tmp_path / "raw"
    raw.mkdir()
    write_csv_rows(unique, [
        {"run_id": "run", "match_id": "MATCH-A", "platform_region": "ME1", "regional_route": "EUROPE"},
        {"run_id": "run", "match_id": "MATCH-A", "platform_region": "ME1", "regional_route": "EUROPE"},
        {"run_id": "run", "match_id": "MATCH-B", "platform_region": "ME1", "regional_route": "EUROPE"},
    ], ["run_id", "match_id", "platform_region", "regional_route"])
    config = {
        "sampling": {"random_seed": 4275},
        "scope": {
            "regional_route": "EUROPE", "study_patch": "17.8",
            "allowed_client_patches": ["16.15"], "set_number": 17,
            "allowed_queue_ids": [1100], "allowed_game_types": ["standard"],
            "allowed_end_results": ["GameComplete"], "expected_participants": 8,
            "start_time": 1_700_000_000, "end_time": 1_800_000_000,
        },
        "validation": {
            "reject_on_wrong_set": True, "reject_on_wrong_patch": True,
            "reject_on_wrong_queue": True, "reject_on_wrong_game_type": True,
            "reject_on_wrong_end_result": True, "reject_outside_collection_window": True,
            "reject_on_unexpected_participant_count": True,
            "reject_on_invalid_participant": True, "require_complete_placement_set": True,
            "require_nonempty_units": True, "require_traits_list": True,
            "require_metadata_participant_match": True,
        },
        "paths": {
            "unique_match_ids": str(unique), "collection_status": str(status),
            "raw_matches": str(raw),
        },
    }
    client = MatchClient()
    rows = collect_matches(config, client, run_id="run", accepted_target=2)
    assert sorted(client.calls) == ["MATCH-A", "MATCH-B"]
    assert len(rows) == 2
    assert sum(str(row["validation_accepted"]).lower() == "true" for row in rows) == 2


def test_existing_verified_is_checksum_checked_revalidated_and_never_inherits_acceptance(
    tmp_path: Path,
) -> None:
    unique = tmp_path / "unique.csv"; status = tmp_path / "status.csv"
    raw = tmp_path / "raw"; raw.mkdir()
    payload = valid_match("MATCH-A")
    payload["info"]["game_version"] = "Linux Version 16.14.1 <Releases/16.14>"
    atomic_json_dump(payload, raw / "MATCH-A.json")
    digest = sha256_file(raw / "MATCH-A.json")
    write_csv_rows(unique, [{"run_id": "v4", "match_id": "MATCH-A", "platform_region": "ME1",
                             "regional_route": "EUROPE"}],
                   ["run_id", "match_id", "platform_region", "regional_route"])
    fields = [
        "run_id", "match_id", "status", "raw_file", "raw_sha256", "first_collected_at",
        "last_checked_at", "validation_accepted", "validation_reasons", "error_class",
    ]
    write_csv_rows(status, [{
        "run_id": "prior", "match_id": "MATCH-A", "status": "downloaded",
        "raw_file": "raw/MATCH-A.json", "raw_sha256": digest,
        "first_collected_at": "before", "last_checked_at": "before",
        "validation_accepted": True, "validation_reasons": [], "error_class": "",
    }], fields)
    client = MatchClient()
    rows = collect_matches(collection_config(unique, status, raw), client, run_id="v4")
    current = next(row for row in rows if row["run_id"] == "v4")
    assert client.calls == []
    assert current["status"] == "existing_verified"
    assert current["validation_accepted"] is False
    assert "wrong_or_missing_client_patch" in current["validation_reasons"]


def test_existing_raw_checksum_mismatch_aborts_without_download(tmp_path: Path) -> None:
    unique = tmp_path / "unique.csv"; status = tmp_path / "status.csv"
    raw = tmp_path / "raw"; raw.mkdir()
    atomic_json_dump(valid_match("MATCH-A"), raw / "MATCH-A.json")
    write_csv_rows(unique, [{"run_id": "v4", "match_id": "MATCH-A", "platform_region": "ME1",
                             "regional_route": "EUROPE"}],
                   ["run_id", "match_id", "platform_region", "regional_route"])
    fields = [
        "run_id", "match_id", "status", "raw_file", "raw_sha256", "first_collected_at",
        "last_checked_at", "validation_accepted", "validation_reasons", "error_class",
    ]
    write_csv_rows(status, [{
        "run_id": "prior", "match_id": "MATCH-A", "status": "downloaded",
        "raw_file": "raw/MATCH-A.json", "raw_sha256": "0" * 64,
        "first_collected_at": "before", "last_checked_at": "before",
        "validation_accepted": True, "validation_reasons": [], "error_class": "",
    }], fields)
    client = MatchClient()
    with pytest.raises(RuntimeError, match="Collection incomplete"):
        collect_matches(collection_config(unique, status, raw), client, run_id="v4")
    assert client.calls == []
    current = next(row for row in read_csv_rows(status) if row["run_id"] == "v4")
    assert current["error_class"] == "raw_checksum_mismatch"

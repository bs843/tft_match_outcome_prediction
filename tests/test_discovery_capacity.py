from __future__ import annotations

from pathlib import Path

import pytest

from src.collect_match_ids import collect_match_ids
from src.collect_matches import collect_matches
from src.collect_players import collect_ladder_players
from src.run_pipeline import resolve_run_id
from src.utils import write_csv_rows
from tests.validation.helpers import valid_match


class LadderClient:
    def __init__(self) -> None:
        self.pages: list[int] = []

    def get_league_entries_by_division(self, platform, tier, division, *, page, queue):
        self.pages.append(page)
        return [{"puuid": f"player-{page}", "tier": tier, "rank": division,
                 "queueType": queue}]


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


class MatchClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_match(self, route, match_id):
        self.calls.append(match_id)
        return valid_match(match_id)


def test_two_ladder_pages_are_honoured(monkeypatch) -> None:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "k" * 32)
    client = LadderClient()
    config = {
        "scope": {"platform_route": "ME1"},
        "sampling": {
            "queue_type": "RANKED_TFT", "seed_players_per_stratum": 1,
            "random_seed": 4275,
            "rank_strata": [{"tier": "GOLD", "divisions": ["I"], "pages_per_division": 2}],
        },
        "privacy": {"hash_key_env": "PARTICIPANT_HASH_KEY"},
    }
    stats = {}
    rows = collect_ladder_players(config, client, run_id="new-v3-run", stats=stats)
    assert client.pages == [1, 2]
    assert len(rows) == 2
    assert stats["ladder_entries_returned"] == 2
    assert stats["unique_players_selected"] == 2


def test_three_history_pages_deduplicate_candidates(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir()
    seeds = private / "seeds.csv"
    write_csv_rows(seeds, [{
        "run_id": "new-v3-run", "puuid": "private-player", "participant_hash": "hash",
        "sampling_tier": "GOLD", "sampling_division": "I",
    }], ["run_id", "puuid", "participant_hash", "sampling_tier", "sampling_division"])
    config = {"scope": {"regional_route": "EUROPE"}, "paths": {"seed_players": str(seeds)}}
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


def test_failed_v2_run_id_cannot_be_resumed_or_merged() -> None:
    with pytest.raises(RuntimeError, match="cannot be resumed or merged"):
        resolve_run_id("main", "20260820T152821Z-9df0ecbb")


def test_duplicate_candidate_rows_are_downloaded_and_counted_once(tmp_path: Path) -> None:
    unique = tmp_path / "unique.csv"
    status = tmp_path / "status.csv"
    raw = tmp_path / "raw"
    raw.mkdir()
    write_csv_rows(unique, [
        {"run_id": "run", "match_id": "MATCH-A"},
        {"run_id": "run", "match_id": "MATCH-A"},
        {"run_id": "run", "match_id": "MATCH-B"},
    ], ["run_id", "match_id"])
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

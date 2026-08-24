from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from analysis.src.roles import LeakageError, assert_no_leakage, load_roles
from src.collect_match_ids import collect_match_ids
from src.collect_matches import balanced_candidate_order
from src.collect_players import collect_ladder_players
from src.collect_rank_snapshots import collect_rank_snapshots
from src.config import ConfigurationError, _validate_config, load_config, route_for_platform
from src.utils import atomic_json_dump, write_csv_rows


class RegionalLadderClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, int]] = []

    def get_league_entries_by_division(self, platform, tier, division, *, page, queue):
        self.calls.append((platform, tier, division, page))
        return [
            {"puuid": "global-shared", "tier": tier, "rank": division, "queueType": queue},
            {"puuid": f"{platform}-player", "tier": tier, "rank": division, "queueType": queue},
        ]


class RegionalHistoryClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get_match_ids(self, route, puuid, *, start, count, start_time, end_time):
        self.calls.append((route, puuid))
        return ["SHARED_MATCH", f"{puuid}_MATCH"]


class CrossPlatformHistoryClient:
    def get_match_ids(self, route, puuid, *, start, count, start_time, end_time):
        assert route == "EUROPE"
        return ["EUW1_7955249354"]


class OutOfScopeHistoryClient:
    def get_match_ids(self, route, puuid, *, start, count, start_time, end_time):
        return ["EUW1_IN_SCOPE", "TR1_OUT_OF_SCOPE"]


class RegionalRankClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get_league_entries_by_puuid(self, platform, puuid):
        self.calls.append((platform, puuid))
        return [{"queueType": "RANKED_TFT", "tier": "GOLD", "rank": "I"}]


def two_region_config() -> dict:
    return {
        "scope": {"platform_route": "ME1", "regional_route": "EUROPE"},
        "platforms": [
            {"platform_region": "ME1", "regional_route": "EUROPE", "enabled": True,
             "seed_players_per_page": 10, "ladder_pages_per_division": 1, "apex_pages": 1},
            {"platform_region": "NA1", "regional_route": "AMERICAS", "enabled": True,
             "seed_players_per_page": 10, "ladder_pages_per_division": 1, "apex_pages": 1},
        ],
        "sampling": {"queue_type": "RANKED_TFT", "seed_players_per_stratum": 10,
                     "random_seed": 4275,
                     "rank_strata": [{"tier": "GOLD", "divisions": ["I"],
                                      "pages_per_division": 1}]},
        "privacy": {"hash_key_env": "PARTICIPANT_HASH_KEY"},
    }


def test_platform_mapping_and_frozen_gold_master_contract() -> None:
    config = load_config()
    assert route_for_platform(config, "ME1") == "EUROPE"
    assert route_for_platform(config, "NA1") == "AMERICAS"
    assert route_for_platform(config, "KR") == "ASIA"
    assert [row["tier"] for row in config["sampling"]["rank_strata"]] == [
        "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER"
    ]
    broken = deepcopy(config)
    broken["platforms"][3]["regional_route"] = "EUROPE"
    with pytest.raises(ConfigurationError, match="Unsupported platform"):
        _validate_config(broken)


def test_balanced_multi_region_ladder_discovery_and_global_player_dedup(monkeypatch) -> None:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "m" * 32)
    client = RegionalLadderClient()
    region_stats: list[dict] = []
    rows = collect_ladder_players(two_region_config(), client, run_id="run",
                                  regional_stats=region_stats)
    assert [call[0] for call in client.calls] == ["ME1", "NA1"]
    assert len(rows) == 3
    assert len({row["puuid"] for row in rows}) == 3
    assert {row["platform_region"] for row in rows} == {"ME1", "NA1"}
    assert {row["regional_route"] for row in rows} == {"EUROPE", "AMERICAS"}
    assert {row["platform_region"] for row in region_stats} == {"ME1", "NA1"}


def test_history_routing_global_candidate_dedup_and_overlap(tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.csv"
    write_csv_rows(seeds, [
        {"run_id": "run", "puuid": "me-player", "participant_hash": "me-hash",
         "platform_region": "ME1", "regional_route": "EUROPE",
         "sampling_tier": "GOLD", "sampling_division": "I", "source": "ranked_ladder"},
        {"run_id": "run", "puuid": "na-player", "participant_hash": "na-hash",
         "platform_region": "NA1", "regional_route": "AMERICAS",
         "sampling_tier": "GOLD", "sampling_division": "I", "source": "ranked_ladder"},
    ], ["run_id", "puuid", "participant_hash", "platform_region", "regional_route",
        "sampling_tier", "sampling_division", "source"])
    client = RegionalHistoryClient()
    stats: dict = {}
    sources, unique = collect_match_ids(
        {"paths": {"seed_players": str(seeds)},
         "sampling": {"candidate_pool_capacity_target": 25000}},
        client, run_id="run", page_size=100, pages_per_seed=1,
        start_time=1, end_time=2, stats=stats,
    )
    assert client.calls == [("EUROPE", "me-player"), ("AMERICAS", "na-player")]
    assert len(sources) == 4
    assert len(unique) == 3
    shared = next(row for row in unique if row["match_id"] == "SHARED_MATCH")
    assert set(shared["source_platform_regions"]) == {"ME1", "NA1"}
    assert stats["cross_region_overlap_candidates"] == 1


def test_match_id_prefix_overrides_seed_platform_for_download_routing(tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.csv"
    write_csv_rows(seeds, [
        {"run_id": "run", "puuid": "transferred-player", "participant_hash": "hash",
         "platform_region": "ME1", "regional_route": "EUROPE",
         "sampling_tier": "GOLD", "sampling_division": "I", "source": "ranked_ladder"},
    ], ["run_id", "puuid", "participant_hash", "platform_region", "regional_route",
        "sampling_tier", "sampling_division", "source"])
    config = two_region_config()
    config["platforms"].insert(1, {
        "platform_region": "EUW1", "regional_route": "EUROPE", "enabled": True,
        "seed_players_per_page": 10, "ladder_pages_per_division": 1, "apex_pages": 1,
    })
    config["paths"] = {"seed_players": str(seeds)}
    _, unique = collect_match_ids(
        config, CrossPlatformHistoryClient(), run_id="run", page_size=100,
        pages_per_seed=1, start_time=1, end_time=2,
    )
    assert unique[0]["platform_region"] == "EUW1"
    assert unique[0]["regional_route"] == "EUROPE"
    assert unique[0]["source_platform_regions"] == ["ME1"]


def test_match_prefix_outside_frozen_platforms_is_reported_and_excluded(tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.csv"
    write_csv_rows(seeds, [
        {"run_id": "run", "puuid": "player", "participant_hash": "hash",
         "platform_region": "ME1", "regional_route": "EUROPE"},
    ], ["run_id", "puuid", "participant_hash", "platform_region", "regional_route"])
    config = two_region_config()
    config["platforms"].append({
        "platform_region": "EUW1", "regional_route": "EUROPE", "enabled": True,
        "seed_players_per_page": 10, "ladder_pages_per_division": 1, "apex_pages": 1,
    })
    config["paths"] = {"seed_players": str(seeds)}
    config["sampling"]["candidate_pool_capacity_target"] = 1
    stats: dict = {}
    sources, unique = collect_match_ids(
        config, OutOfScopeHistoryClient(), run_id="run", page_size=100,
        pages_per_seed=1, start_time=1, end_time=2, stats=stats,
    )
    assert {row["match_id"] for row in sources} == {"EUW1_IN_SCOPE", "TR1_OUT_OF_SCOPE"}
    assert [row["match_id"] for row in unique] == ["EUW1_IN_SCOPE"]
    assert stats["out_of_scope_platform_candidate_ids"] == 1
    assert stats["unique_candidate_match_ids"] == 1
    assert stats["capacity_gate_result"] == "pass"


def test_balanced_candidate_order_interleaves_regions() -> None:
    rows = [
        {"match_id": f"ME1_{index}", "platform_region": "ME1"} for index in range(3)
    ] + [{"match_id": f"NA1_{index}", "platform_region": "NA1"} for index in range(3)]
    ordered = balanced_candidate_order(rows, 4275)
    assert [row["platform_region"] for row in ordered] == ["ME1", "NA1"] * 3


def test_rank_snapshots_use_each_accepted_match_platform(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "r" * 32)
    raw = tmp_path / "raw"; raw.mkdir()
    accepted = tmp_path / "accepted.csv"
    rank = tmp_path / "rank.csv"
    write_csv_rows(accepted, [
        {"run_id": "run", "match_id": "ME1_A", "platform_region": "ME1", "regional_route": "EUROPE"},
        {"run_id": "run", "match_id": "NA1_A", "platform_region": "NA1", "regional_route": "AMERICAS"},
    ], ["run_id", "match_id", "platform_region", "regional_route"])
    atomic_json_dump({"info": {"game_datetime": 1000,
                                "participants": [{"puuid": "me-player"}]}}, raw / "ME1_A.json")
    atomic_json_dump({"info": {"game_datetime": 1000,
                                "participants": [{"puuid": "na-player"}]}}, raw / "NA1_A.json")
    config = {"sampling": {"queue_type": "RANKED_TFT"},
              "privacy": {"hash_key_env": "PARTICIPANT_HASH_KEY"},
              "paths": {"accepted_match_ids": str(accepted), "raw_matches": str(raw),
                        "rank_snapshots": str(rank),
                        "rank_snapshot_routing": str(tmp_path / "routing.csv")}}
    client = RegionalRankClient()
    rows = collect_rank_snapshots(config, client, run_id="run")
    assert client.calls == [("ME1", "me-player"), ("NA1", "na-player")]
    assert {(row["platform_region"], row["regional_route"]) for row in rows} == {
        ("ME1", "EUROPE"), ("NA1", "AMERICAS")
    }


def test_region_is_provenance_and_blocked_from_primary_features() -> None:
    rules = load_roles(Path("analysis/configs/columns.yaml"))
    with pytest.raises(LeakageError):
        assert_no_leakage(["level", "platform_region"], rules)

from __future__ import annotations

from pathlib import Path

import pytest

from src.api_client import RiotAPIError, RiotForbiddenError, RiotNotFoundError
from src.collect_rank_snapshots import collect_rank_snapshots, resolve_rank_routing
from src.utils import atomic_json_dump, read_csv_rows, sha256_file, write_csv_rows


class SnapshotClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls: list[str] = []

    def get_league_entries_by_puuid(self, platform_route: str, puuid: str):
        self.calls.append(puuid)
        response = self.responses[puuid]
        if isinstance(response, Exception):
            raise response
        return response


def snapshot_config(tmp_path: Path) -> dict:
    return {
        "scope": {"platform_route": "ME1"},
        "sampling": {"queue_type": "RANKED_TFT"},
        "privacy": {"hash_key_env": "PARTICIPANT_HASH_KEY"},
        "paths": {
            "accepted_match_ids": str(tmp_path / "accepted.csv"),
            "raw_matches": str(tmp_path / "raw"),
            "rank_snapshots": str(tmp_path / "rank_snapshots.csv"),
            "rank_snapshot_routing": str(tmp_path / "rank_snapshot_routing.csv"),
        },
    }


def prepare_match(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    write_csv_rows(
        tmp_path / "accepted.csv",
        [{"run_id": "run-1", "match_id": "ME1_TEST", "platform_region": "ME1",
          "regional_route": "EUROPE"}],
        ["run_id", "match_id", "platform_region", "regional_route"],
    )
    atomic_json_dump({
        "info": {"game_datetime": 1000,
                 "participants": [{"puuid": f"player-{index}"} for index in range(1, 5)]}
    }, tmp_path / "raw" / "ME1_TEST.json")


def test_every_participant_gets_explicit_rank_status(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "k" * 32)
    prepare_match(tmp_path)
    client = SnapshotClient({
        "player-1": [{
            "queueType": "RANKED_TFT", "tier": "GOLD", "rank": "II",
            "leaguePoints": 44, "wins": 12, "losses": 9,
        }],
        "player-2": [{"queueType": "RANKED_TFT_DOUBLE_UP", "tier": "SILVER"}],
        "player-3": RiotNotFoundError("missing"),
        "player-4": RiotAPIError("temporary failure"),
    })

    rows = collect_rank_snapshots(snapshot_config(tmp_path), client, run_id="run-1")

    assert {row["rank_snapshot_status"] for row in rows} == {
        "found", "unranked", "not_found", "request_failed",
    }
    assert all(row["rank_snapshot_queue_type"] == "RANKED_TFT" for row in rows)
    output = (tmp_path / "rank_snapshots.csv").read_text(encoding="utf-8")
    assert "player-" not in output
    assert "temporary failure" not in output


def test_resume_retries_only_request_failures(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "k" * 32)
    prepare_match(tmp_path)
    first = SnapshotClient({
        "player-1": [], "player-2": [], "player-3": [],
        "player-4": RiotAPIError("temporary failure"),
    })
    collect_rank_snapshots(snapshot_config(tmp_path), first, run_id="run-1")
    second = SnapshotClient({
        "player-4": [{
            "queueType": "RANKED_TFT", "tier": "PLATINUM", "rank": "IV",
            "leaguePoints": 1, "wins": 1, "losses": 1,
        }],
    })

    rows = collect_rank_snapshots(
        snapshot_config(tmp_path), second, run_id="run-1", retry_failed=True
    )

    assert second.calls == ["player-4"]
    statuses = [row["rank_snapshot_status"] for row in rows]
    assert statuses.count("found") == 1
    assert statuses.count("unranked") == 3
    stored = read_csv_rows(tmp_path / "rank_snapshots.csv")
    assert len(stored) == 4


def prepare_platform_history(tmp_path: Path, observations: list[tuple[str, str, int]]) -> dict:
    raw = tmp_path / "raw"
    raw.mkdir()
    accepted_rows = []
    route = {"ME1": "EUROPE", "EUW1": "EUROPE"}
    for match_id, platform, timestamp in observations:
        accepted_rows.append({
            "run_id": "run", "match_id": match_id,
            "platform_region": platform, "regional_route": route[platform],
        })
        atomic_json_dump({
            "info": {
                "game_datetime": timestamp,
                "participants": [{"puuid": "migrating-player"}],
            }
        }, raw / f"{match_id}.json")
    accepted = tmp_path / "accepted.csv"
    write_csv_rows(
        accepted, accepted_rows,
        ["run_id", "match_id", "platform_region", "regional_route"],
    )
    return {
        "privacy": {"hash_key_env": "PARTICIPANT_HASH_KEY"},
        "paths": {"accepted_match_ids": str(accepted), "raw_matches": str(raw)},
    }


@pytest.mark.parametrize("platform", ["ME1", "EUW1"])
def test_single_platform_routes_to_that_platform(monkeypatch, tmp_path: Path, platform: str) -> None:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "s" * 32)
    config = prepare_platform_history(tmp_path, [(f"{platform}_A", platform, 1000)])
    identities, routing = resolve_rank_routing(config, "run")
    assert len(identities) == len(routing) == 1
    assert routing[0]["selected_platform_region"] == platform
    assert routing[0]["platform_transfer_detected"] is False


def test_me1_to_euw1_selects_latest_and_retains_provenance(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "s" * 32)
    config = prepare_platform_history(tmp_path, [
        ("ME1_A", "ME1", 1000), ("ME1_B", "ME1", 2000),
        ("EUW1_C", "EUW1", 3000),
    ])
    identities, routing = resolve_rank_routing(config, "run")
    assert len(identities) == len(routing) == 1
    assert routing[0]["selected_platform_region"] == "EUW1"
    assert routing[0]["observed_platforms"] == ["EUW1", "ME1"]
    assert routing[0]["accepted_match_occurrence_count"] == 3
    assert routing[0]["platform_transfer_detected"] is True


def test_euw1_to_me1_selects_latest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "s" * 32)
    config = prepare_platform_history(tmp_path, [
        ("EUW1_A", "EUW1", 1000), ("ME1_B", "ME1", 2000),
    ])
    _, routing = resolve_rank_routing(config, "run")
    assert routing[0]["selected_platform_region"] == "ME1"


def test_same_platform_duplicates_produce_one_routing_row(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "s" * 32)
    config = prepare_platform_history(tmp_path, [
        ("ME1_A", "ME1", 1000), ("ME1_B", "ME1", 2000),
    ])
    identities, routing = resolve_rank_routing(config, "run")
    assert len(identities) == len(routing) == 1
    assert routing[0]["accepted_match_occurrence_count"] == 2


def test_equal_timestamp_uses_lexicographically_greatest_match_id(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "s" * 32)
    config = prepare_platform_history(tmp_path, [
        ("EUW1_A", "EUW1", 1000), ("ME1_Z", "ME1", 1000),
    ])
    _, routing = resolve_rank_routing(config, "run")
    assert routing[0]["selected_from_match_id"] == "ME1_Z"
    assert routing[0]["selected_platform_region"] == "ME1"


def test_routing_is_identical_for_identical_frozen_inputs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "s" * 32)
    config = prepare_platform_history(tmp_path, [
        ("ME1_A", "ME1", 1000), ("EUW1_B", "EUW1", 2000),
    ])
    assert resolve_rank_routing(config, "run")[1] == resolve_rank_routing(config, "run")[1]


def test_rank_request_failure_never_changes_accepted_cohort(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "k" * 32)
    prepare_match(tmp_path)
    before = sha256_file(tmp_path / "accepted.csv")
    client = SnapshotClient({
        f"player-{index}": RiotAPIError("failure") for index in range(1, 5)
    })
    rows = collect_rank_snapshots(snapshot_config(tmp_path), client, run_id="run-1")
    assert all(row["rank_snapshot_status"] == "request_failed" for row in rows)
    assert sha256_file(tmp_path / "accepted.csv") == before


def test_resource_forbidden_is_persisted_as_request_failed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "k" * 32)
    prepare_match(tmp_path)
    client = SnapshotClient({
        f"player-{index}": RiotForbiddenError("resource denied")
        for index in range(1, 5)
    })
    rows = collect_rank_snapshots(snapshot_config(tmp_path), client, run_id="run-1")
    assert all(row["rank_snapshot_status"] == "request_failed" for row in rows)
    assert all(row["rank_snapshot_error_class"] == "RiotForbiddenError" for row in rows)

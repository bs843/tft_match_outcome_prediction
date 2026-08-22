from __future__ import annotations

from pathlib import Path

from src.api_client import RiotAPIError, RiotNotFoundError
from src.collect_rank_snapshots import collect_rank_snapshots
from src.utils import atomic_json_dump, read_csv_rows, write_csv_rows


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
        "info": {"participants": [{"puuid": f"player-{index}"} for index in range(1, 5)]}
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

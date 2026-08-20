from __future__ import annotations

from pathlib import Path

from src.create_splits import create_temporal_splits, split_manifest_path
from src.utils import write_csv_rows


def test_temporal_splits_keep_matches_intact_and_report_player_overlap(tmp_path: Path) -> None:
    interim, splits, reports = tmp_path / "interim", tmp_path / "splits", tmp_path / "reports"
    interim.mkdir()
    match_rows = [{"match_hash": f"m{i}", "game_datetime_ms": i} for i in range(10)]
    participant_rows = [
        {"match_hash": f"m{i}", "participant_hash": f"p{i}-{j}"}
        for i in range(10) for j in range(8)
    ]
    write_csv_rows(interim / "matches.csv", match_rows, ["match_hash", "game_datetime_ms"])
    write_csv_rows(interim / "participants.csv", participant_rows, ["match_hash", "participant_hash"])
    config = {
        "project": {"dataset_version": "test"},
        "splits": {"train_fraction": 0.7, "validation_fraction": 0.15, "test_fraction": 0.15},
        "paths": {"interim": str(interim), "splits": str(splits), "reports": str(reports)},
    }
    rows, report = create_temporal_splits(config)
    assert report["counts"] == {"train": 7, "validation": 1, "test": 2}
    assert report["match_overlap"] == 0
    assert len({row["match_hash"] for row in rows}) == 10
    assert split_manifest_path(config, "run-1").name == "match_splits_test.csv"

from __future__ import annotations

from pathlib import Path

from src.build_features import build_feature_matrix
from src.create_splits import create_temporal_splits
from src.normalise_matches import (
    AUGMENT_FIELDS, MATCH_FIELDS, PARTICIPANT_FIELDS, TRAIT_FIELDS, UNIT_FIELDS,
    UNIT_ITEM_FIELDS, normalise_matches,
)
from src.collect_rank_snapshots import RANK_SNAPSHOT_FIELDS
from src.utils import atomic_json_dump, hash_identifier, write_csv_rows


def test_raw_to_features_is_pseudonymous_and_train_vocabulary_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PARTICIPANT_HASH_KEY", "x" * 32)
    raw, private, interim, splits, reports, processed = [tmp_path / name for name in
                                                         ("raw", "private", "interim", "splits", "reports", "processed")]
    for path in (raw, private, interim, splits, reports, processed):
        path.mkdir()
    config = {
        "project": {"dataset_version": "test-v1", "schema_version": "1.0.0"},
        "scope": {"study_patch": "17.8"},
        "privacy": {"hash_key_env": "PARTICIPANT_HASH_KEY"},
        "splits": {"train_fraction": 0.7, "validation_fraction": 0.15, "test_fraction": 0.15},
        "features": {"minimum_category_frequency": 1},
        "paths": {
            "accepted_match_ids": str(private / "accepted.csv"), "raw_matches": str(raw),
            "rank_snapshots": str(private / "rank_snapshots.csv"),
            "interim": str(interim), "splits": str(splits), "reports": str(reports),
            "processed": str(processed),
        },
    }
    accepted = []
    for match_index in range(10):
        match_id = f"ME1_PRIVATE_{match_index}"
        accepted.append({"run_id": "run-1", "match_id": match_id})
        participants = []
        for placement in range(1, 9):
            augment = "TEST_ONLY_AUGMENT" if match_index == 9 else "TRAIN_AUGMENT"
            participants.append({
                "puuid": f"private-player-{match_index}-{placement}", "placement": placement,
                "level": 8, "gold_left": 2, "last_round": 30,
                "players_eliminated": 1, "total_damage_to_players": 50,
                "time_eliminated": 1000, "augments": [augment],
                "units": [{"character_id": "TFT17_UNIT", "rarity": 2, "tier": 2,
                           "items": [1], "itemNames": ["TFT17_ITEM"]}],
                "traits": [{"name": "TFT17_TRAIT", "num_units": 2, "style": 1,
                            "tier_current": 1, "tier_total": 3}],
            })
        atomic_json_dump({
            "metadata": {"match_id": match_id, "participants": [p["puuid"] for p in participants]},
            "info": {"game_datetime": (1_750_000_000 + match_index) * 1000,
                     "game_length": 1800, "queue_id": 1100, "tft_game_type": "standard",
                     "tft_set_number": 17, "participants": participants},
        }, raw / f"{match_id}.json")
    write_csv_rows(private / "accepted.csv", accepted, ["run_id", "match_id"])
    snapshot_rows = []
    for match_index in range(10):
        for placement in range(1, 9):
            snapshot_rows.append({
                "run_id": "run-1",
                "participant_hash": hash_identifier(
                    f"private-player-{match_index}-{placement}", "x" * 32
                ),
                "rank_snapshot_tier": "GOLD",
                "rank_snapshot_division": "II",
                "rank_snapshot_league_points": 42,
                "rank_snapshot_wins": 10,
                "rank_snapshot_losses": 8,
                "rank_snapshot_queue_type": "RANKED_TFT",
                "rank_snapshot_retrieved_at": "2026-08-10T00:00:00Z",
                "rank_snapshot_status": "found",
                "rank_snapshot_error_class": "",
            })
    write_csv_rows(private / "rank_snapshots.csv", snapshot_rows, RANK_SNAPSHOT_FIELDS)

    tables = normalise_matches(config, run_id="run-1")
    schemas = {
        "matches": MATCH_FIELDS, "participants": PARTICIPANT_FIELDS, "units": UNIT_FIELDS,
        "unit_items": UNIT_ITEM_FIELDS, "traits": TRAIT_FIELDS, "augments": AUGMENT_FIELDS,
    }
    for name, fields in schemas.items():
        write_csv_rows(interim / f"{name}.csv", tables[name], fields)
    assert "private-player" not in (interim / "participants.csv").read_text(encoding="utf-8")
    assert "ME1_PRIVATE" not in (interim / "matches.csv").read_text(encoding="utf-8")
    assert {row["rank_snapshot_status"] for row in tables["participants"]} == {"found"}

    split_rows, _ = create_temporal_splits(config)
    write_csv_rows(splits / "match_splits.csv", split_rows,
                   ["match_hash", "split", "game_datetime_ms", "dataset_version"])
    rows, fields, vocabulary = build_feature_matrix(config)
    assert len(rows) == 80
    assert "TRAIN_AUGMENT" in vocabulary["augment"]
    assert "TEST_ONLY_AUGMENT" not in vocabulary["augment"]
    assert not {
        "last_round", "placement_group", "participant_hash", "rank_snapshot_tier",
        "rank_snapshot_status",
    }.intersection(
        set(fields) - {"participant_hash"}
    )

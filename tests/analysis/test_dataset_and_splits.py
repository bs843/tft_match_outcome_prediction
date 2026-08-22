from __future__ import annotations

import pandas as pd
import pytest

from analysis.src.dataset import (DatasetIntegrityError, participant_rows_from_payload,
                                  validate_participant_dataset)
from analysis.src.splits import create_match_split, cv_splits
from analysis.src.synthetic import synthetic_participant_dataset


def test_participant_integrity_and_top4_derivation() -> None:
    frame = synthetic_participant_dataset(5)
    validate_participant_dataset(frame)
    broken = frame.copy()
    broken.loc[0, "top4"] = 0
    with pytest.raises(DatasetIntegrityError, match="top4"):
        validate_participant_dataset(broken)


def test_participant_extraction_constructs_targets_and_clean_rank() -> None:
    participants = []
    for placement in range(1, 9):
        participants.append({
            "participant_hash": f"p{placement}", "placement": placement, "level": 8,
            "units": [{"character_id": "UnitA", "tier": 2,
                       "itemNames": ["ItemA"]}],
            "traits": [{"name": "TraitA", "num_units": 2, "style": 1,
                        "tier_current": 1}],
        })
    payload = {
        "metadata": {"match_id": "safe-match"},
        "info": {"queue_id": 1100, "tft_set_number": 17, "game_datetime": 1,
                 "participants": participants},
        "validation": {"match_hash": "safe-match", "status": "valid",
                       "platform_region": "ME1", "regional_route": "EUROPE"},
    }
    rows = participant_rows_from_payload(
        payload, "run", {"p1": {"rank_tier": "DIAMOND", "rank_division": "I",
                                  "rank_status": "found"}}, study_patch="17.8",
    )
    assert len(rows) == 8
    assert rows[0]["top4"] == 1 and rows[-1]["top4"] == 0
    assert rows[0]["rank_tier"] == "DIAMOND"
    assert not any(key.startswith("rank_snapshot_") for key in rows[0])


def test_manifest_row_reconciliation_fails_closed() -> None:
    frame = synthetic_participant_dataset(2)
    approval = type("Approval", (), {"accepted_count": 3, "participant_count": 24})()
    with pytest.raises(DatasetIntegrityError, match="approved cohort"):
        validate_participant_dataset(frame, approval)


def test_split_is_deterministic_and_match_disjoint() -> None:
    frame = synthetic_participant_dataset(20)
    first = create_match_split(frame, seed=4275, test_fraction=0.2)
    second = create_match_split(frame.sample(frac=1, random_state=8), seed=4275, test_fraction=0.2)
    assert first.checksum == second.checksum
    assert set(first.development.match_hash).isdisjoint(set(first.test.match_hash))
    assert first.test.match_hash.nunique() == 4


def test_group_cross_validation_has_no_match_overlap() -> None:
    frame = synthetic_participant_dataset(20)
    split = create_match_split(frame, seed=4275, test_fraction=0.2)
    groups = split.development.match_hash.reset_index(drop=True)
    for train, validation in cv_splits(split.development.reset_index(drop=True), folds=4, seed=4275):
        assert set(groups.iloc[train]).isdisjoint(set(groups.iloc[validation]))

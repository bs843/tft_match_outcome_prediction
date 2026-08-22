from __future__ import annotations

import pandas as pd
import pytest

from analysis.src.features import CompositionFeatureEncoder
from analysis.src.roles import LeakageError, assert_no_leakage, load_roles
from analysis.src.synthetic import synthetic_participant_dataset


def test_encoder_is_deterministic_and_schema_stable() -> None:
    frame = synthetic_participant_dataset(6)
    first = CompositionFeatureEncoder(2).fit(frame)
    second = CompositionFeatureEncoder(2).fit(frame.sample(frac=1, random_state=9))
    assert first.vocabulary_metadata() == second.vocabulary_metadata()
    transformed = first.transform(frame, ["basic", "units", "traits", "items", "structure"])
    assert len(transformed) == len(frame)
    assert transformed.isna().sum().sum() == 0
    assert list(transformed) == list(first.transform(frame.iloc[:3],
                                                     ["basic", "units", "traits", "items", "structure"]))


def test_encoder_vocabulary_is_frozen_after_development_fit() -> None:
    development = synthetic_participant_dataset(4)
    encoder = CompositionFeatureEncoder(1).fit(development)
    test = development.iloc[:1].copy()
    test.loc[test.index[0], "units_json"] = '[{"name":"UNSEEN","stars":3,"items":[]}]'
    columns = encoder.transform(test, ["units"]).columns
    assert not any("unseen" in column for column in columns)


def test_leakage_checker_blocks_targets_identifiers_and_rank_snapshots() -> None:
    rules = load_roles(__import__("pathlib").Path("analysis/configs/columns.yaml"))
    for prohibited in ("placement", "top4", "match_hash", "participant_hash",
                       "rank_snapshot_tier", "validation_status", "source_sha256"):
        with pytest.raises(LeakageError):
            assert_no_leakage(["level", prohibited], rules)
    assert_no_leakage(["level", "unit_presence__safe"], rules)

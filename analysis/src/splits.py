from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .common import stable_hash


@dataclass(frozen=True)
class MatchSplit:
    assignments: pd.DataFrame
    development: pd.DataFrame
    test: pd.DataFrame
    checksum: str


def create_match_split(frame: pd.DataFrame, *, seed: int, test_fraction: float) -> MatchSplit:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    match_ids = sorted(set(map(str, frame["match_hash"])))
    if len(match_ids) < 2:
        raise ValueError("At least two matches are required for a split")
    shuffled = match_ids.copy()
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * test_fraction))
    test_ids = set(shuffled[:test_count])
    rows = [{"match_hash": match_id,
             "split": "test" if match_id in test_ids else "development"}
            for match_id in match_ids]
    assignments = pd.DataFrame(rows)
    development = frame.loc[~frame["match_hash"].isin(test_ids)].copy()
    test = frame.loc[frame["match_hash"].isin(test_ids)].copy()
    assert_disjoint_groups(development, test)
    checksum = stable_hash(rows)
    return MatchSplit(assignments, development, test, checksum)


def assert_disjoint_groups(development: pd.DataFrame, test: pd.DataFrame) -> None:
    overlap = set(development["match_hash"]).intersection(test["match_hash"])
    if overlap:
        raise RuntimeError(f"Match leakage across development/test: {len(overlap)} matches")


def cv_splits(frame: pd.DataFrame, *, folds: int, seed: int):
    from sklearn.model_selection import StratifiedGroupKFold

    groups = frame["match_hash"].astype(str)
    labels = frame["top4"].astype(int)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_indices, validation_indices in splitter.split(frame, labels, groups):
        train_groups = set(groups.iloc[train_indices])
        validation_groups = set(groups.iloc[validation_indices])
        if train_groups.intersection(validation_groups):
            raise RuntimeError("Match leakage across cross-validation folds")
        yield train_indices, validation_indices

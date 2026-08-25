from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .common import read_csv, stable_hash


class FrozenSplitError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenMatchSplit:
    assignments: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    checksum: str


def canonical_split_hash(rows: list[dict[str, str]]) -> str:
    canonical = sorted(
        ({"match_hash": str(row.get("match_hash", "")),
          "split": str(row.get("split", ""))} for row in rows),
        key=lambda row: row["match_hash"],
    )
    return stable_hash(canonical)


def validate_frozen_assignments(
    rows: list[dict[str, str]], *, expected_match_ids: set[str],
    expected_counts: dict[str, int], dataset_version: str,
) -> str:
    allowed = {"train", "validation", "test"}
    if len(rows) != sum(expected_counts.values()):
        raise FrozenSplitError("Frozen split row count differs from the analysis contract")
    ids = [str(row.get("match_hash", "")) for row in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise FrozenSplitError("Frozen split contains blank or duplicate match identifiers")
    observed = {name: sum(row.get("split") == name for row in rows) for name in allowed}
    if observed != expected_counts:
        raise FrozenSplitError(
            f"Frozen split counts differ: expected {expected_counts}, observed {observed}"
        )
    invalid = {str(row.get("split", "")) for row in rows}.difference(allowed)
    if invalid:
        raise FrozenSplitError(f"Frozen split contains invalid partitions: {sorted(invalid)}")
    if any(str(row.get("dataset_version", "")) != dataset_version for row in rows):
        raise FrozenSplitError("Frozen split contains an unexpected dataset version")
    observed_ids = set(ids)
    missing = expected_match_ids.difference(observed_ids)
    unexpected = observed_ids.difference(expected_match_ids)
    if missing or unexpected:
        raise FrozenSplitError(
            f"Frozen split/cohort mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return canonical_split_hash(rows)


def load_frozen_match_split(
    frame: pd.DataFrame, manifest: Path, *, expected_counts: dict[str, int],
    dataset_version: str, expected_hash: str,
) -> FrozenMatchSplit:
    if not manifest.is_file():
        raise FrozenSplitError(f"Authoritative frozen split is missing: {manifest}")
    rows = read_csv(manifest)
    frame_ids = set(frame["match_hash"].astype(str))
    checksum = validate_frozen_assignments(
        rows, expected_match_ids=frame_ids, expected_counts=expected_counts,
        dataset_version=dataset_version,
    )
    if checksum != expected_hash:
        raise FrozenSplitError(
            f"Frozen split hash mismatch: expected {expected_hash}, observed {checksum}"
        )
    assignments = pd.DataFrame([
        {"match_hash": row["match_hash"], "split": row["split"]} for row in rows
    ]).sort_values("match_hash", kind="stable").reset_index(drop=True)
    joined = frame.merge(assignments, on="match_hash", how="left", validate="many_to_one")
    if joined["split"].isna().any():
        raise FrozenSplitError("Participant rows contain matches absent from the frozen split")
    if not (joined.groupby("match_hash")["split"].nunique() == 1).all():
        raise FrozenSplitError("Participants from one match cross split partitions")
    partitions = {
        name: joined.loc[joined["split"] == name].drop(columns="split").copy()
        for name in ("train", "validation", "test")
    }
    assert_disjoint_groups(partitions["train"], partitions["validation"], partitions["test"])
    return FrozenMatchSplit(assignments, partitions["train"], partitions["validation"],
                            partitions["test"], checksum)


def assert_disjoint_groups(*frames: pd.DataFrame) -> None:
    groups = [set(frame["match_hash"].astype(str)) for frame in frames]
    for index, left in enumerate(groups):
        for right in groups[index + 1:]:
            if left.intersection(right):
                raise FrozenSplitError("Match leakage across frozen partitions")


def create_match_split(frame: pd.DataFrame, *, seed: int, test_fraction: float):
    """Deprecated engineering-only helper; production analysis never calls it."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    match_ids = sorted(set(map(str, frame["match_hash"])))
    shuffled = match_ids.copy()
    random.Random(seed).shuffle(shuffled)
    test_ids = set(shuffled[:max(1, round(len(shuffled) * test_fraction))])
    assignments = pd.DataFrame([
        {"match_hash": match_id, "split": "test" if match_id in test_ids else "development"}
        for match_id in match_ids
    ])
    development = frame.loc[~frame["match_hash"].isin(test_ids)].copy()
    test = frame.loc[frame["match_hash"].isin(test_ids)].copy()
    return type("LegacyEngineeringSplit", (), {
        "assignments": assignments, "development": development, "test": test,
        "checksum": stable_hash(assignments.to_dict("records")),
    })()


def cv_splits(frame: pd.DataFrame, *, folds: int, seed: int):
    from sklearn.model_selection import StratifiedGroupKFold

    groups = frame["match_hash"].astype(str)
    labels = frame["top4"].astype(int)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_indices, validation_indices in splitter.split(frame, labels, groups):
        train_groups = set(groups.iloc[train_indices])
        validation_groups = set(groups.iloc[validation_indices])
        if train_groups.intersection(validation_groups):
            raise FrozenSplitError("Match leakage across cross-validation folds")
        yield train_indices, validation_indices

from __future__ import annotations

import json

import pandas as pd

from analysis.src.features import CompositionFeatureEncoder
from analysis.thesis_extension.finalize import match_cluster_ci
from analysis.thesis_extension.relations import connected_match_components
from analysis.thesis_extension.run import FAMILIES, GROUPS, _combine, feature_group


def _frame(prefix: str) -> pd.DataFrame:
    return pd.DataFrame({
        "level": [7, 8], "composition_size": [7, 8],
        "units_json": [json.dumps([{"name": f"{prefix}_a", "stars": 2}]),
                       json.dumps([{"name": f"{prefix}_b", "stars": 1}])],
        "traits_json": [json.dumps([{"name": "trait_a", "tier_current": 1}]),
                        json.dumps([{"name": "trait_b", "tier_current": 0}])],
        "items_json": [json.dumps(["item_a"]), json.dumps(["item_b"])],
    })


def test_feature_families_are_cumulative() -> None:
    previous: set[str] = set()
    for family in ("F1", "F2", "F3", "F4", "F5"):
        current = set(FAMILIES[family])
        assert previous.issubset(current)
        previous = current
    assert tuple(FAMILIES["F5"]) == GROUPS


def test_encoder_vocabulary_is_fold_local_and_unseen_safe() -> None:
    fit = pd.concat([_frame("fit")] * 3, ignore_index=True)
    held = _frame("held")
    encoder = CompositionFeatureEncoder(minimum_frequency=1).fit(fit)
    fit_x = encoder.transform(fit, GROUPS)
    held_x = encoder.transform(held, GROUPS)
    assert list(fit_x.columns) == list(held_x.columns)
    assert not any("held" in column for column in held_x.columns)


def test_substantive_group_mapping_is_exhaustive_for_contract_patterns() -> None:
    expected = {
        "level": "basic", "composition_size": "basic",
        "unit_presence__a": "units", "unit_star__a": "units",
        "trait_presence__a": "traits", "trait_strength__a": "traits",
        "item_count__a": "items", "structure__mean_star": "structure",
    }
    assert {name: feature_group(name) for name in expected} == expected


def test_combining_groups_does_not_mutate_inputs() -> None:
    matrices = {group: pd.DataFrame({group: [1.0, 2.0]}) for group in GROUPS}
    combined = _combine(("basic", "items"), matrices)
    assert list(combined.columns) == ["basic", "items"]
    assert all(list(frame.columns) == [group] for group, frame in matrices.items())


def test_cluster_bootstrap_resamples_complete_matches() -> None:
    frame = pd.DataFrame({
        "match_hash": ["a"] * 8 + ["b"] * 8,
        "truth": [0, 0, 0, 0, 1, 1, 1, 1] * 2,
        "estimate": [0, 0, 0, 0, 1, 1, 1, 1] * 2,
    })
    result = match_cluster_ci(frame, {"accuracy": lambda y, p: float((y == p).mean())},
                              samples=10, seed=4275)
    assert result.loc[0, "estimate"] == 1.0
    assert result.loc[0, "resampling_unit"] == "match"
    assert result.loc[0, "bootstrap_samples"] == 10


def test_player_connected_components_join_matches_transitively() -> None:
    frame = pd.DataFrame({
        "match_hash": ["a", "a", "b", "b", "c", "c"],
        "participant_hash": ["p1", "p2", "p2", "p3", "p3", "p4"],
    })
    components = connected_match_components(frame)
    assert components.nunique() == 1

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from .common import safe_feature_name
from .roles import assert_no_leakage


BASIC = ("level", "composition_size")
STRUCTURE = (
    "structure__unique_units", "structure__mean_star", "structure__max_star",
    "structure__two_star_units", "structure__three_star_units",
    "structure__total_items", "structure__unique_items",
    "structure__active_traits", "structure__trait_strength_total",
)


@dataclass
class CompositionFeatureEncoder:
    minimum_frequency: int = 5
    unit_vocabulary: tuple[str, ...] = field(default_factory=tuple)
    trait_vocabulary: tuple[str, ...] = field(default_factory=tuple)
    item_vocabulary: tuple[str, ...] = field(default_factory=tuple)
    fitted: bool = False

    def fit(self, frame: pd.DataFrame) -> "CompositionFeatureEncoder":
        if self.fitted:
            raise RuntimeError("Feature encoder is immutable after fitting")
        unit_counts: Counter[str] = Counter()
        trait_counts: Counter[str] = Counter()
        item_counts: Counter[str] = Counter()
        for _, row in frame.iterrows():
            unit_counts.update({unit["name"] for unit in _json_list(row["units_json"])})
            trait_counts.update({trait["name"] for trait in _json_list(row["traits_json"])})
            item_counts.update(set(_json_list(row["items_json"])))
        self.unit_vocabulary = _vocabulary(unit_counts, self.minimum_frequency)
        self.trait_vocabulary = _vocabulary(trait_counts, self.minimum_frequency)
        self.item_vocabulary = _vocabulary(item_counts, self.minimum_frequency)
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame, groups: Iterable[str]) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Fit the feature encoder on development data first")
        group_set = set(groups)
        unknown = group_set.difference({"basic", "units", "traits", "items", "structure"})
        if unknown:
            raise ValueError(f"Unknown feature group(s): {sorted(unknown)}")
        rows: list[dict[str, float]] = []
        for _, source in frame.iterrows():
            units = _json_list(source["units_json"])
            traits = _json_list(source["traits_json"])
            items = _json_list(source["items_json"])
            row: dict[str, float] = {}
            if "basic" in group_set:
                row.update({"level": float(source["level"]),
                            "composition_size": float(source["composition_size"])})
            if "units" in group_set:
                by_name = {str(unit["name"]): int(unit.get("stars", 0)) for unit in units}
                for name in self.unit_vocabulary:
                    suffix = safe_feature_name(name)
                    row[f"unit_presence__{suffix}"] = float(name in by_name)
                    row[f"unit_star__{suffix}"] = float(by_name.get(name, 0))
            if "traits" in group_set:
                by_name = {str(trait["name"]): trait for trait in traits}
                for name in self.trait_vocabulary:
                    suffix = safe_feature_name(name)
                    trait = by_name.get(name, {})
                    row[f"trait_presence__{suffix}"] = float(name in by_name)
                    row[f"trait_strength__{suffix}"] = float(
                        trait.get("tier_current", trait.get("style", 0)) or 0
                    )
            if "items" in group_set:
                counts = Counter(map(str, items))
                for name in self.item_vocabulary:
                    row[f"item_count__{safe_feature_name(name)}"] = float(counts.get(name, 0))
            if "structure" in group_set:
                stars = [int(unit.get("stars", 0)) for unit in units]
                active = [trait for trait in traits if int(trait.get("tier_current", 0) or 0) > 0]
                row.update({
                    "structure__unique_units": float(len({unit["name"] for unit in units})),
                    "structure__mean_star": float(sum(stars) / len(stars)) if stars else 0.0,
                    "structure__max_star": float(max(stars, default=0)),
                    "structure__two_star_units": float(sum(star == 2 for star in stars)),
                    "structure__three_star_units": float(sum(star >= 3 for star in stars)),
                    "structure__total_items": float(len(items)),
                    "structure__unique_items": float(len(set(map(str, items)))),
                    "structure__active_traits": float(len(active)),
                    "structure__trait_strength_total": float(sum(
                        int(trait.get("tier_current", 0) or 0) for trait in active
                    )),
                })
            rows.append(row)
        return pd.DataFrame(rows, index=frame.index).fillna(0.0).astype(float)

    def fit_transform(self, frame: pd.DataFrame, groups: Iterable[str]) -> pd.DataFrame:
        return self.fit(frame).transform(frame, groups)

    def vocabulary_metadata(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("Feature encoder has not been fitted")
        return {
            "minimum_frequency": self.minimum_frequency,
            "unit_vocabulary": list(self.unit_vocabulary),
            "trait_vocabulary": list(self.trait_vocabulary),
            "item_vocabulary": list(self.item_vocabulary),
        }


def build_feature_families(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    experiments: dict[str, Any],
    column_rules: dict[str, Any],
) -> tuple[CompositionFeatureEncoder, dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    encoder = CompositionFeatureEncoder(int(experiments["minimum_category_frequency"]))
    # Scientific preprocessing is fitted on the frozen train partition only.
    # Validation is transform-only; the test partition is not accepted here.
    encoder.fit(train)
    result = {}
    for family, details in experiments["feature_families"].items():
        groups = details["groups"]
        train_x = encoder.transform(train, groups)
        validation_x = encoder.transform(validation, groups)
        if list(train_x.columns) != list(validation_x.columns):
            raise RuntimeError(f"{family} train/validation feature schemas differ")
        assert_no_leakage(train_x.columns, column_rules)
        result[family] = (train_x, validation_x)
    return encoder, result


def transform_feature_families(
    encoder: CompositionFeatureEncoder, frame: pd.DataFrame,
    experiments: dict[str, Any], column_rules: dict,
) -> dict[str, pd.DataFrame]:
    """Transform a held-out partition using an already frozen train encoder."""
    result: dict[str, pd.DataFrame] = {}
    for family, details in experiments["feature_families"].items():
        matrix = encoder.transform(frame, details["groups"])
        assert_no_leakage(matrix.columns, column_rules)
        result[family] = matrix
    return result


def validate_matrix_contract(
    columns: list[str], *, family: str, contract: dict[str, Any],
) -> None:
    features = contract["features"]
    groups = set(features["families"][family])
    required: set[str] = set()
    for group in groups:
        required.update(features.get("required_exact_by_group", {}).get(group, []))
    missing = required.difference(columns)
    if missing:
        raise RuntimeError(f"{family} is missing contracted features: {sorted(missing)}")
    allowed_exact = set().union(*(
        set(values) for values in features.get("required_exact_by_group", {}).values()
    ))
    prefixes = tuple(features["allowed_dynamic_prefixes"])
    if any(column.startswith("rank_snapshot_") for column in columns):
        raise RuntimeError("Rank snapshot fields entered a production feature matrix")
    unexpected = [column for column in columns
                  if column not in allowed_exact and not column.startswith(prefixes)]
    if unexpected:
        raise RuntimeError(f"{family} contains unexpected production features: {unexpected}")


def _vocabulary(counts: Counter[str], minimum: int) -> tuple[str, ...]:
    return tuple(sorted(name for name, count in counts.items() if count >= minimum))


def _json_list(value: Any) -> list:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise ValueError("Serialized feature collection must be a JSON list")
    return parsed

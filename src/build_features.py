from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from typing import Any

from .config import ensure_project_directories, load_config, project_path
from .create_splits import split_manifest_path
from .utils import atomic_json_dump, read_csv_rows, safe_filename, safe_float, write_csv_rows

ID_FIELDS = ["match_hash", "participant_hash", "split", "top_four", "placement"]
AGGREGATE_FIELDS = [
    "board_size", "unique_units", "mean_star_level", "max_star_level",
    "total_items", "unique_items", "active_traits", "augment_count",
]


def build_feature_matrix(
    config: dict[str, Any], *, run_id: str | None = None
) -> tuple[list[dict[str, Any]], list[str], dict[str, list[str]]]:
    interim = project_path(config, "interim") / run_id if run_id else project_path(config, "interim")
    participants = read_csv_rows(interim / "participants.csv",
                                 ["match_hash", "participant_hash", "top_four", "placement"])
    split_path = split_manifest_path(config, run_id)
    if run_id and not split_path.exists():
        legacy_path = project_path(config, "splits") / run_id / "match_splits.csv"
        if legacy_path.exists():
            split_path = legacy_path
    splits = {row["match_hash"]: row["split"] for row in
              read_csv_rows(split_path,
                            ["match_hash", "split"])}
    key = lambda row: (row["match_hash"], row["participant_hash"])
    units = read_csv_rows(interim / "units.csv")
    items = read_csv_rows(interim / "unit_items.csv")
    traits = read_csv_rows(interim / "traits.csv")
    augments = read_csv_rows(interim / "augments.csv")

    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    records_by_family = {
        "unit": [(key(row), row.get("character_id", "")) for row in units],
        "item": [(key(row), row.get("item_name") or row.get("item_id", "")) for row in items],
        "trait": [(key(row), row.get("trait_id", "")) for row in traits if row.get("trait_active") == "true"],
        "augment": [(key(row), row.get("augment_id", "")) for row in augments],
    }
    for family, records in records_by_family.items():
        for participant_key, value in records:
            if value and splits.get(participant_key[0]) == "train":
                category_counts[family][value] += 1
    minimum = int(config["features"]["minimum_category_frequency"])
    vocabulary = {
        family: sorted(value for value, count in counts.items() if count >= minimum)
        for family, counts in category_counts.items()
    }
    allowed = {family: set(values) for family, values in vocabulary.items()}
    dynamic_fields = [_feature_name(family, value) for family in sorted(vocabulary)
                      for value in vocabulary[family]]

    by_participant: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    stars: dict[tuple[str, str], list[float]] = defaultdict(list)
    unique_units: dict[tuple[str, str], set[str]] = defaultdict(set)
    unique_items: dict[tuple[str, str], set[str]] = defaultdict(set)
    item_counts: Counter[tuple[str, str]] = Counter()
    active_trait_counts: Counter[tuple[str, str]] = Counter()
    augment_counts: Counter[tuple[str, str]] = Counter()
    for row in units:
        participant_key, value = key(row), row.get("character_id", "")
        if value in allowed["unit"]:
            by_participant[participant_key][_feature_name("unit", value)] += 1
        if value:
            unique_units[participant_key].add(value)
        stars[participant_key].append(safe_float(row.get("star_level"), 0) or 0)
    for row in items:
        participant_key = key(row)
        value = row.get("item_name") or row.get("item_id", "")
        if value in allowed["item"]:
            by_participant[participant_key][_feature_name("item", value)] += 1
        if value:
            unique_items[participant_key].add(value)
            item_counts[participant_key] += 1
    for row in traits:
        participant_key, value = key(row), row.get("trait_id", "")
        if row.get("trait_active") == "true" and value in allowed["trait"]:
            by_participant[participant_key][_feature_name("trait", value)] += 1
        if row.get("trait_active") == "true":
            active_trait_counts[participant_key] += 1
    for row in augments:
        participant_key, value = key(row), row.get("augment_id", "")
        if value in allowed["augment"]:
            by_participant[participant_key][_feature_name("augment", value)] += 1
        if value:
            augment_counts[participant_key] += 1

    rows: list[dict[str, Any]] = []
    for participant in participants:
        participant_key = key(participant)
        star_values = stars[participant_key]
        dynamic = by_participant[participant_key]
        row: dict[str, Any] = {
            "match_hash": participant["match_hash"], "participant_hash": participant["participant_hash"],
            "split": splits[participant["match_hash"]], "top_four": participant["top_four"],
            "placement": participant["placement"], "board_size": len(star_values),
            "unique_units": len(unique_units[participant_key]),
            "mean_star_level": sum(star_values) / len(star_values) if star_values else 0,
            "max_star_level": max(star_values, default=0),
            "total_items": item_counts[participant_key],
            "unique_items": len(unique_items[participant_key]),
            "active_traits": active_trait_counts[participant_key],
            "augment_count": augment_counts[participant_key],
        }
        row.update({field: dynamic.get(field, 0) for field in dynamic_fields})
        rows.append(row)
    return rows, ID_FIELDS + AGGREGATE_FIELDS + dynamic_fields, vocabulary


def _feature_name(family: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{family}__{safe_filename(value)[:80]}__{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-vocabulary composition features")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    rows, fields, vocabulary = build_feature_matrix(config, run_id=args.run_id)
    output_dir = project_path(config, "processed") / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "features.csv", rows, fields)
    atomic_json_dump(vocabulary, output_dir / "feature_vocabulary.json")
    print(f"Built {len(rows)} rows with {len(fields) - len(ID_FIELDS)} features")


if __name__ == "__main__":
    main()

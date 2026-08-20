from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from .config import ensure_project_directories, load_config, project_path
from .utils import atomic_json_dump, read_csv_rows, safe_filename, write_csv_rows

SPLIT_FIELDS = ["match_hash", "split", "game_datetime_ms", "dataset_version"]


def split_manifest_path(config: dict[str, Any], run_id: str | None = None):
    base = project_path(config, "splits") / run_id if run_id else project_path(config, "splits")
    if not run_id:
        return base / "match_splits.csv"
    version = safe_filename(str(config["project"]["dataset_version"]))
    return base / f"match_splits_{version}.csv"


def create_temporal_splits(
    config: dict[str, Any], *, run_id: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    interim = project_path(config, "interim") / run_id if run_id else project_path(config, "interim")
    matches = read_csv_rows(interim / "matches.csv", ["match_hash", "game_datetime_ms"])
    participants = read_csv_rows(interim / "participants.csv", ["match_hash", "participant_hash"])
    if len(matches) < 3:
        raise RuntimeError("At least three matches are required for train/validation/test splits")
    ordered = sorted(matches, key=lambda row: (int(row["game_datetime_ms"]), row["match_hash"]))
    n = len(ordered)
    train_end = max(1, int(n * float(config["splits"]["train_fraction"])))
    validation_size = max(1, int(n * float(config["splits"]["validation_fraction"])))
    validation_end = min(n - 1, train_end + validation_size)
    if validation_end <= train_end:
        validation_end = train_end + 1
    assignment: dict[str, str] = {}
    for index, row in enumerate(ordered):
        split = "train" if index < train_end else "validation" if index < validation_end else "test"
        assignment[row["match_hash"]] = split
    rows = [{
        "match_hash": row["match_hash"], "split": assignment[row["match_hash"]],
        "game_datetime_ms": row["game_datetime_ms"],
        "dataset_version": config["project"]["dataset_version"],
    } for row in ordered]

    players: dict[str, set[str]] = defaultdict(set)
    for row in participants:
        players[assignment[row["match_hash"]]].add(row["participant_hash"])
    overlap = {
        "train_validation": len(players["train"] & players["validation"]),
        "train_test": len(players["train"] & players["test"]),
        "validation_test": len(players["validation"] & players["test"]),
    }
    report = {
        "run_id": run_id,
        "strategy": "chronological_match_group",
        "counts": {name: sum(row["split"] == name for row in rows)
                   for name in ("train", "validation", "test")},
        "participant_overlap": overlap,
        "match_overlap": 0,
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze leakage-safe temporal match splits")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    rows, report = create_temporal_splits(config, run_id=args.run_id)
    output = split_manifest_path(config, args.run_id)
    output_dir = output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = read_csv_rows(output, ["match_hash", "split"])
        if existing != [{key: str(row.get(key, "")) for key in SPLIT_FIELDS} for row in rows]:
            raise RuntimeError("Frozen split manifest already exists with different assignments")
    else:
        write_csv_rows(output, rows, SPLIT_FIELDS)
    version = safe_filename(str(config["project"]["dataset_version"]))
    atomic_json_dump(
        {**report, "dataset_version": config["project"]["dataset_version"]},
        project_path(config, "reports") / f"split_audit_{args.run_id}_{version}.json",
    )
    print(f"Frozen split counts: {report['counts']}")


if __name__ == "__main__":
    main()

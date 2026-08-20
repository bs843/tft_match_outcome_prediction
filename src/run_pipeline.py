from __future__ import annotations

import argparse
import subprocess
import sys

from .config import ensure_project_directories, load_config, project_path
from .collection_report import write_collection_report
from .provenance import (
    accepted_manifest_checksum,
    initialise_run_manifest,
    new_run_id,
    update_run_manifest,
    verify_frozen_git_state,
)
from .utils import read_csv_rows


class CandidateCapacityError(RuntimeError):
    """Raised before match download when discovery cannot support the main target."""


def run_module(module: str, *arguments: str) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print("\n$ " + " ".join(command))
    subprocess.run(command, check=True)


def resolve_run_id(target: str, requested_run_id: str | None) -> str:
    if target == "main" and requested_run_id:
        raise RuntimeError(
            "A main collection must start with a new automatically generated run ID; "
            "failed or interrupted final runs cannot be resumed or merged"
        )
    return requested_run_id or new_run_id()


def enforce_candidate_capacity(config: dict, run_id: str) -> int:
    unique_count = len({
        row.get("match_id", "") for row in read_csv_rows(project_path(config, "unique_match_ids"))
        if row.get("run_id") == run_id and row.get("match_id", "")
    })
    threshold = int(config["sampling"]["candidate_pool_capacity_target"])
    if unique_count < threshold:
        raise CandidateCapacityError(
            f"Candidate capacity gate failed: {unique_count} unique candidates; "
            f"at least {threshold} required before match download"
        )
    print(f"Candidate capacity gate passed: {unique_count} >= {threshold}")
    return unique_count


def run_acquisition_stages(
    config: dict, run_common: list[str], *, run_id: str, player_mode: str,
    accepted_target: int, enforce_capacity: bool,
) -> None:
    run_module("src.collect_players", *run_common, "--mode", player_mode)
    run_module("src.collect_match_ids", *run_common)
    if enforce_capacity:
        enforce_candidate_capacity(config, run_id)
    run_module("src.collect_matches", *run_common, "--accepted-target", str(accepted_target))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible patch-17.8 MSc pipeline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--player-mode", choices=["ladder", "manual"], default="ladder")
    parser.add_argument("--target", choices=["pilot", "main"], default="pilot")
    parser.add_argument("--train", action="store_true", help="Train after freezing the dataset and splits")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_project_directories(config)
    if config["scope"].get("start_time") is None or config["scope"].get("end_time") is None:
        raise RuntimeError("Set verified scope.start_time and scope.end_time before any API collection")
    accepted_target = int(config["sampling"][
        "pilot_accepted_matches" if args.target == "pilot" else "target_accepted_matches"
    ])
    git_state = None
    if args.target == "main" and config["provenance"]["enforce_git_integrity_for_main"]:
        git_state = verify_frozen_git_state(config)
    run_id = resolve_run_id(args.target, args.run_id)
    manifest = initialise_run_manifest(
        config, run_id, git_state=git_state,
        requested_accepted_match_target=accepted_target,
    )
    common = (["--config", args.config] if args.config else [])
    run_common = [*common, "--run-id", run_id]
    try:
        run_acquisition_stages(
            config, run_common, run_id=run_id, player_mode=args.player_mode,
            accepted_target=accepted_target, enforce_capacity=args.target == "main",
        )
        run_module("src.validate_matches", *run_common, "--accepted-target", str(accepted_target))
        accepted = len([row for row in read_csv_rows(project_path(config, "accepted_match_ids"))
                        if row.get("run_id") == run_id])
        if accepted != accepted_target:
            raise RuntimeError(f"Expected exactly {accepted_target} accepted matches, received {accepted}")
        run_module("src.collect_rank_snapshots", *run_common)
        run_module("src.normalise_matches", *run_common)
        run_module("src.create_splits", *run_common)
        run_module("src.build_features", *run_common)
        if args.train:
            run_module("src.train_model", *run_common, "--experiment-id", run_id)
        update_run_manifest(manifest, status="complete", stage_counts={
            "accepted_matches": accepted,
            "participants": accepted * int(config["scope"]["expected_participants"]),
        }, metadata={
            "accepted_match_manifest_id": f"accepted_match_ids.csv#run_id={run_id}",
            "accepted_match_manifest_sha256": accepted_manifest_checksum(
                run_id, read_csv_rows(project_path(config, "accepted_match_ids"))
            ),
        })
        write_collection_report(config, run_id)
    except CandidateCapacityError:
        update_run_manifest(manifest, status="capacity_failed")
        write_collection_report(config, run_id)
        raise
    except Exception:
        update_run_manifest(manifest, status="failed")
        write_collection_report(config, run_id)
        raise
    print(f"\nPipeline complete. Run ID: {run_id}")


if __name__ == "__main__":
    main()

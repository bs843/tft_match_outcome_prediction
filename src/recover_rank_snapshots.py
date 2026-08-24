from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .collection_report import write_collection_report
from .collect_rank_snapshots import (
    RANK_ROUTING_VERSION, RANK_SNAPSHOT_STATUSES, resolve_rank_routing,
)
from .config import ensure_project_directories, load_config, project_path
from .provenance import (
    accepted_manifest_checksum, config_hash, verify_frozen_git_state,
)
from .utils import (
    atomic_json_dump, load_json, read_csv_rows, safe_filename, sha256_file,
    stable_json_hash, utc_now_iso,
)
from .validation.validate_collection import run_validation

RECOVERY_STAGE_MODULES = (
    "src.collect_rank_snapshots",
    "src.normalise_matches",
    "src.create_splits",
    "src.build_features",
)
ORIGINAL_FAILURE_REASON = (
    "collect_rank_snapshots rejected a participant observed on both ME1 and EUW1 "
    "under the superseded one-platform invariant"
)


class RankRecoveryError(RuntimeError):
    """Raised when downstream recovery cannot prove cohort immutability."""


def cohort_fingerprint(
    config: dict[str, Any], run_id: str, *, verify_raw: bool = True,
) -> dict[str, Any]:
    accepted_path = project_path(config, "accepted_match_ids")
    all_rows = read_csv_rows(accepted_path)
    rows = [row for row in all_rows if row.get("run_id") == run_id]
    match_ids = [row.get("match_id", "") for row in rows]
    fingerprint = {
        "accepted_match_count": len(rows),
        "unique_accepted_match_count": len(set(match_ids)),
        "accepted_manifest_sha256": accepted_manifest_checksum(run_id, rows),
        "accepted_match_id_set_sha256": stable_json_hash(sorted(match_ids)),
        "accepted_file_sha256": sha256_file(accepted_path),
        "all_rows_validation_accepted": all(
            row.get("validation_accepted", "").lower() == "true" for row in rows
        ),
        "raw_checksum_failures": 0,
    }
    if verify_raw:
        raw_dir = project_path(config, "raw_matches")
        failures = 0
        for row in rows:
            raw_path = raw_dir / f"{safe_filename(row.get('match_id', ''))}.json"
            if (not raw_path.is_file() or not row.get("raw_sha256") or
                    sha256_file(raw_path) != row.get("raw_sha256")):
                failures += 1
        fingerprint["raw_checksum_failures"] = failures
    return fingerprint


def validate_recovery_authorization(
    config: dict[str, Any], run_id: str, *, verify_raw: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    authorization = config["downstream_recovery"]
    if run_id != authorization["authorized_run_id"]:
        raise RankRecoveryError("Run ID is not authorized for downstream-only recovery")
    manifest_path = project_path(config, "manifests") / f"collection_run_{run_id}.json"
    if not manifest_path.is_file():
        raise RankRecoveryError("Original collection manifest is missing")
    original_manifest = load_json(manifest_path)
    if original_manifest.get("status") != "failed":
        raise RankRecoveryError("Recovery requires the preserved failed collection manifest")
    if original_manifest.get("configuration_version") != authorization["original_configuration_version"]:
        raise RankRecoveryError("Original collection configuration provenance is unexpected")
    if int(original_manifest.get("requested_accepted_match_target") or 0) != 5000:
        raise RankRecoveryError("Original run did not request exactly 5,000 accepted matches")

    fingerprint = cohort_fingerprint(config, run_id, verify_raw=verify_raw)
    expected = {
        "accepted_match_count": int(authorization["expected_accepted_matches"]),
        "unique_accepted_match_count": int(authorization["expected_accepted_matches"]),
        "accepted_manifest_sha256": authorization["accepted_manifest_sha256"],
        "accepted_match_id_set_sha256": authorization["accepted_match_id_set_sha256"],
        "accepted_file_sha256": authorization["accepted_file_sha256"],
        "all_rows_validation_accepted": True,
        "raw_checksum_failures": 0,
    }
    if fingerprint != expected:
        raise RankRecoveryError(
            f"Accepted cohort integrity gate failed: observed={fingerprint} expected={expected}"
        )

    identities, routing_rows = resolve_rank_routing(config, run_id)
    if len(identities) != int(authorization["expected_unique_participants"]):
        raise RankRecoveryError("Unique participant count differs from recovery authorization")
    if len(routing_rows) != len(identities):
        raise RankRecoveryError("Rank routing is not one-to-one with participants")
    if len({row["participant_hash"] for row in routing_rows}) != len(routing_rows):
        raise RankRecoveryError("Rank routing contains duplicate participant rows")
    transfers = [row for row in routing_rows if row["platform_transfer_detected"]]
    if len(transfers) != int(authorization["expected_multi_platform_participants"]):
        raise RankRecoveryError("Multi-platform participant count differs from authorization")
    if any(row["routing_version"] != RANK_ROUTING_VERSION for row in routing_rows):
        raise RankRecoveryError("Unexpected rank-routing version")
    return original_manifest, fingerprint, routing_rows


def routing_summary(
    config: dict[str, Any], run_id: str, fingerprint: dict[str, Any],
    routing_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    transfers = [row for row in routing_rows if row["platform_transfer_detected"]]
    return {
        "run_id": run_id,
        "amendment_version": RANK_ROUTING_VERSION,
        "dataset_version": config["project"]["dataset_version"],
        "schema_version": config["project"]["schema_version"],
        "accepted_cohort": fingerprint,
        "unique_participants": len(routing_rows),
        "multi_platform_participants": len(transfers),
        "routing_decisions_changed": len(transfers),
        "selected_platform_counts": dict(sorted(Counter(
            row["selected_platform_region"] for row in routing_rows
        ).items())),
        "transfer_decisions": [{
            "participant_hash": row["participant_hash"],
            "selected_platform_region": row["selected_platform_region"],
            "selected_from_match_id": row["selected_from_match_id"],
            "selected_match_timestamp_ms": row["selected_match_timestamp_ms"],
            "observed_platforms": row["observed_platforms"],
        } for row in transfers],
        "candidate_or_match_collection_invoked": False,
    }


def _run_stage(module: str, run_id: str) -> None:
    if module not in RECOVERY_STAGE_MODULES:
        raise RankRecoveryError(f"Module {module} is not allowed in downstream recovery")
    command = [sys.executable, "-m", module, "--run-id", run_id]
    if module == "src.collect_rank_snapshots":
        command.append("--retry-failed")
    print("\n$ " + " ".join(command))
    subprocess.run(command, check=True)


def _assert_rank_and_feature_outputs(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    expected = int(config["downstream_recovery"]["expected_unique_participants"])
    rank_rows = [row for row in read_csv_rows(project_path(config, "rank_snapshots"))
                 if row.get("run_id") == run_id]
    routing_rows = [row for row in read_csv_rows(project_path(config, "rank_snapshot_routing"))
                    if row.get("run_id") == run_id]
    if len(rank_rows) != expected or len({row.get("participant_hash") for row in rank_rows}) != expected:
        raise RankRecoveryError("Rank snapshots are not exactly one row per participant")
    if len(routing_rows) != expected or len({row.get("participant_hash") for row in routing_rows}) != expected:
        raise RankRecoveryError("Rank routing provenance is not exactly one row per participant")
    invalid = {row.get("rank_snapshot_status", "") for row in rank_rows}.difference(
        RANK_SNAPSHOT_STATUSES
    )
    if invalid:
        raise RankRecoveryError(f"Invalid rank snapshot statuses: {sorted(invalid)}")
    features = project_path(config, "processed") / run_id / "features.csv"
    with features.open("r", encoding="utf-8-sig", newline="") as handle:
        fields = next(csv.reader(handle), [])
    prohibited = set(config["features"]["prohibited_columns"])
    leaked = [field for field in fields if field in prohibited or field.startswith("rank_snapshot_")]
    if leaked:
        raise RankRecoveryError(f"Rank/leakage fields entered primary features: {leaked}")
    return {
        "rank_snapshot_count": len(rank_rows),
        "rank_status_counts": dict(sorted(Counter(
            row["rank_snapshot_status"] for row in rank_rows
        ).items())),
        "rank_failures": sum(
            row["rank_snapshot_status"] == "request_failed" for row in rank_rows
        ),
        "routing_provenance_count": len(routing_rows),
        "primary_feature_columns_checked": len(fields),
    }


def _proposed_completed_manifest(
    config: dict[str, Any], run_id: str, original: dict[str, Any],
    git_state: dict[str, str], fingerprint: dict[str, Any], recovery_path: Path,
    output_counts: dict[str, Any], transfer_count: int,
) -> dict[str, Any]:
    return {
        **original,
        "status": "complete",
        "completed_at": utc_now_iso(),
        "config_hash": config_hash(config),
        "dataset_version": config["project"]["dataset_version"],
        "schema_version": config["project"]["schema_version"],
        "configuration_version": config["provenance"]["configuration_version"],
        "git_commit": git_state["git_commit"],
        "git_tag": git_state["git_tag"],
        "accepted_match_manifest_id": f"accepted_match_ids.csv#run_id={run_id}",
        "accepted_match_manifest_sha256": fingerprint["accepted_manifest_sha256"],
        "stage_counts": {
            **original.get("stage_counts", {}),
            "accepted_matches": fingerprint["accepted_match_count"],
            "participants": (
                fingerprint["accepted_match_count"] *
                int(config["scope"]["expected_participants"])
            ),
            "unique_participants": output_counts["rank_snapshot_count"],
            "rank_snapshots": output_counts["rank_snapshot_count"],
        },
        "acquisition_provenance": {
            "status_after_original_failure": original.get("status"),
            "configuration_version": original.get("configuration_version"),
            "dataset_version": original.get("dataset_version"),
            "schema_version": original.get("schema_version"),
            "git_commit": original.get("git_commit"),
            "git_tag": original.get("git_tag"),
            "config_hash": original.get("config_hash"),
        },
        "failure_history": [{
            "status": original.get("status"),
            "completed_at": original.get("completed_at"),
            "reason": ORIGINAL_FAILURE_REASON,
        }],
        "downstream_recovery": {
            "amendment_version": RANK_ROUTING_VERSION,
            "recovered_at": utc_now_iso(),
            "recovery_manifest": str(recovery_path.name),
            "multi_platform_participants": transfer_count,
            "routing_decisions_changed": transfer_count,
            "accepted_cohort_unchanged": True,
        },
    }


def recover(config: dict[str, Any], run_id: str, *, dry_run: bool) -> Path:
    git_state = verify_frozen_git_state(config)
    original, before, routing_rows = validate_recovery_authorization(config, run_id)
    summary = routing_summary(config, run_id, before, routing_rows)
    report_root = project_path(config, "reports") / "recovery"
    report_root.mkdir(parents=True, exist_ok=True)
    if dry_run:
        path = report_root / f"rank_snapshot_recovery_{run_id}_dry_run.json"
        atomic_json_dump({**summary, "status": "DRY_RUN_PASS", "checked_at": utc_now_iso()}, path)
        print(f"Routing-only recovery dry run passed: {path}")
        return path

    recovery_path = project_path(config, "manifests") / (
        f"rank_snapshot_recovery_{run_id}_{RANK_ROUTING_VERSION}.json"
    )
    if recovery_path.exists() and load_json(recovery_path).get("status") == "complete":
        raise RankRecoveryError("A completed recovery manifest already exists")
    recovery_manifest = {
        **summary,
        "status": "started",
        "started_at": utc_now_iso(),
        "original_manifest": f"collection_run_{run_id}.json",
        "original_manifest_sha256": stable_json_hash(original),
        "original_failure_reason": ORIGINAL_FAILURE_REASON,
        "recovery_command": f"python -m src.recover_rank_snapshots --run-id {run_id}",
        "git_commit": git_state["git_commit"],
        "git_tag": git_state["git_tag"],
    }
    atomic_json_dump(recovery_manifest, recovery_path)
    try:
        for module in RECOVERY_STAGE_MODULES:
            _run_stage(module, run_id)
        output_counts = _assert_rank_and_feature_outputs(config, run_id)
        after = cohort_fingerprint(config, run_id, verify_raw=True)
        if after != before:
            raise RankRecoveryError("Accepted cohort or raw checksums changed during recovery")
        proposed = _proposed_completed_manifest(
            config, run_id, original, git_state, after, recovery_path,
            output_counts, summary["multi_platform_participants"],
        )
        validation_summary, issues = run_validation(
            config, run_id=run_id, manifest_override=proposed,
        )
        if validation_summary.collection_status == "FAIL" or any(
            issue.severity == "critical" for issue in issues
        ):
            raise RankRecoveryError("Post-recovery collection validation failed")
        final_fingerprint = cohort_fingerprint(config, run_id, verify_raw=True)
        if final_fingerprint != before:
            raise RankRecoveryError("Accepted cohort changed during final validation")
        atomic_json_dump(proposed, project_path(config, "manifests") / f"collection_run_{run_id}.json")
        atomic_json_dump({
            **recovery_manifest,
            **output_counts,
            "status": "complete",
            "completed_at": utc_now_iso(),
            "accepted_cohort_after": final_fingerprint,
            "accepted_cohort_unchanged": True,
            "validation_status": validation_summary.collection_status,
        }, recovery_path)
        write_collection_report(config, run_id)
        print(f"Rank-snapshot recovery complete: {recovery_path}")
        return recovery_path
    except BaseException as exc:
        atomic_json_dump({
            **recovery_manifest,
            "status": "failed",
            "completed_at": utc_now_iso(),
            "error_class": type(exc).__name__,
            "accepted_cohort_after": cohort_fingerprint(config, run_id, verify_raw=False),
        }, recovery_path)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover only rank snapshots and downstream artifacts for an authorized cohort"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    recover(config, args.run_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

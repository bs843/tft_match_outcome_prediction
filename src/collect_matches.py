from __future__ import annotations

import argparse
import random
from collections import OrderedDict
from typing import Any

from .api_client import RiotAPIError, RiotApiClient, RiotAuthenticationError, RiotNotFoundError
from .config import ensure_project_directories, load_config, project_path
from .utils import (
    atomic_json_dump, load_json, read_csv_rows, safe_filename, sha256_file,
    utc_now_iso, write_csv_rows,
)
from .validate_matches import validate_match_payload

STATUS_FIELDS = [
    "run_id", "match_id", "status", "raw_file", "raw_sha256", "first_collected_at",
    "last_checked_at", "validation_accepted", "validation_reasons", "error_class",
]


class RawIntegrityError(RuntimeError):
    """Raised when an existing immutable raw response no longer matches provenance."""


def collect_matches(
    config: dict[str, Any], client: RiotApiClient, *, run_id: str,
    accepted_target: int | None = None, candidate_limit: int | None = None,
) -> list[dict[str, Any]]:
    discovered_rows = [row for row in read_csv_rows(project_path(config, "unique_match_ids"),
                                                     ["run_id", "match_id"])
                       if row.get("run_id") == run_id]
    # Defend the accounting/download stage even if its nominally unique input is
    # malformed: one match ID can be evaluated and counted at most once per run.
    match_rows = list(OrderedDict(
        (row.get("match_id", "").strip(), row) for row in discovered_rows
        if row.get("match_id", "").strip()
    ).values())
    if not match_rows:
        raise RuntimeError(f"No candidate match IDs found for run {run_id}")
    rng = random.Random(int(config["sampling"]["random_seed"]))
    rng.shuffle(match_rows)
    if candidate_limit is not None:
        match_rows = match_rows[:max(0, candidate_limit)]

    status_path = project_path(config, "collection_status")
    existing = read_csv_rows(status_path)
    status_map: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict(
        ((row.get("run_id", ""), row.get("match_id", "")), row)
        for row in existing if row.get("run_id") and row.get("match_id")
    )
    known_digests: dict[str, set[str]] = {}
    for row in existing:
        match_id = row.get("match_id", "")
        digest = row.get("raw_sha256", "")
        if match_id and digest:
            known_digests.setdefault(match_id, set()).add(digest)
    raw_dir = project_path(config, "raw_matches")
    regional_route = config["scope"]["regional_route"]
    accepted = 0
    failed = 0

    for index, candidate in enumerate(match_rows, start=1):
        if accepted_target is not None and accepted >= accepted_target:
            break
        match_id = candidate["match_id"].strip()
        raw_path = raw_dir / f"{safe_filename(match_id)}.json"
        key = (run_id, match_id)
        checked_at = utc_now_iso()
        try:
            if raw_path.exists() and raw_path.stat().st_size > 0:
                digest = sha256_file(raw_path)
                if known_digests.get(match_id) and digest not in known_digests[match_id]:
                    raise RawIntegrityError("existing raw checksum disagrees with recorded provenance")
                payload = load_json(raw_path)
                status = "existing_verified"
                first_collected = status_map.get(key, {}).get("first_collected_at", "") or checked_at
            else:
                payload = client.get_match(regional_route, match_id)
                atomic_json_dump(payload, raw_path)
                digest = sha256_file(raw_path)
                status = "downloaded"
                first_collected = checked_at
            previous_digest = status_map.get(key, {}).get("raw_sha256", "")
            if previous_digest and previous_digest != digest:
                raise RuntimeError("immutable raw checksum changed")
            validation = validate_match_payload(
                payload, config, expected_match_id=match_id,
                raw_file=str(raw_path.relative_to(raw_dir.parents[2])), run_id=run_id,
            )
            accepted += int(validation.accepted)
            status_map[key] = {
                "run_id": run_id, "match_id": match_id, "status": status,
                "raw_file": str(raw_path.relative_to(raw_dir.parents[2])), "raw_sha256": digest,
                "first_collected_at": first_collected, "last_checked_at": checked_at,
                "validation_accepted": validation.accepted,
                "validation_reasons": validation.rejection_reasons,
                "error_class": "",
            }
            print(f"{index}/{len(match_rows)} accepted={accepted} {status}")
        except RiotAuthenticationError:
            raise
        except RiotNotFoundError:
            status_map[key] = _failure_row(run_id, match_id, "not_found", checked_at)
        except RawIntegrityError:
            failed += 1
            status_map[key] = _failure_row(run_id, match_id, "raw_checksum_mismatch", checked_at)
        except RiotAPIError as exc:
            failed += 1
            status_map[key] = _failure_row(run_id, match_id, type(exc).__name__, checked_at)
        # Checkpoint in batches. Raw JSON is already durable, so interruption can
        # lose only a few status rows and those files are revalidated on resume.
        if index % 10 == 0:
            write_csv_rows(status_path, status_map.values(), STATUS_FIELDS)

    rows = list(status_map.values())
    write_csv_rows(status_path, rows, STATUS_FIELDS)
    run_rows = [row for row in rows if row["run_id"] == run_id]
    if failed:
        raise RuntimeError(f"Collection incomplete: {failed} candidate requests failed")
    if accepted_target is not None and accepted < accepted_target:
        raise RuntimeError(f"Candidate pool exhausted with {accepted}/{accepted_target} accepted matches")
    return run_rows


def _failure_row(run_id: str, match_id: str, error_class: str, checked_at: str) -> dict[str, Any]:
    return {
        "run_id": run_id, "match_id": match_id, "status": "failed", "raw_file": "",
        "raw_sha256": "", "first_collected_at": "", "last_checked_at": checked_at,
        "validation_accepted": False, "validation_reasons": [], "error_class": error_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download immutable and immediately validated match responses")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--accepted-target", type=int, default=None)
    parser.add_argument("--candidate-limit", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_project_directories(config)
    with RiotApiClient(config) as client:
        rows = collect_matches(config, client, run_id=args.run_id,
                               accepted_target=args.accepted_target,
                               candidate_limit=args.candidate_limit)
    print(f"Recorded {len(rows)} candidate statuses")


if __name__ == "__main__":
    main()

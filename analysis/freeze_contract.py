from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import PROJECT_ROOT, load_config

from .src.common import sha256_file, utc_now, write_json
from .src.contract import (AnalysisContractError, load_contract, run_analysis_preflight,
                           verify_analysis_release)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the immutable TFT analysis-contract release"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-tests", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config(args.config)
    contract = load_contract()
    tag = str(contract["release"]["required_analysis_tag"])
    if not args.skip_tests:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT, check=True)
    run_analysis_preflight(config, verify_release=False)
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise AnalysisContractError(
            "Freeze requires a clean committed worktree after all tests pass"
        )
    if _tag_exists(tag):
        raise AnalysisContractError(f"Analysis release tag already exists and cannot be moved: {tag}")
    _git("tag", "-a", tag, "-m",
         f"Frozen TFT Patch 17.8 analysis contract {contract['analysis_contract_version']}")
    commit, _ = verify_analysis_release(contract)
    release_path = PROJECT_ROOT / contract["release"]["release_manifest"]
    write_json(release_path, {
        "analysis_contract_version": contract["analysis_contract_version"],
        "analysis_tag": tag,
        "analysis_commit": commit,
        "created_at_utc": utc_now(),
        "contract_sha256": sha256_file(Path(contract["_path"])),
        "experiments_sha256": contract["configuration"]["experiments_sha256"],
        "columns_sha256": contract["configuration"]["columns_sha256"],
        "dataset_manifest_sha256": contract["dataset"]["dataset_manifest_sha256"],
        "split_assignment_sha256": contract["split"]["canonical_assignment_sha256"],
        "tests_required_before_release": True,
        "status": "frozen",
    })
    print(f"Frozen analysis contract {contract['analysis_contract_version']} at {tag} ({commit})")
    print(f"Release manifest: {release_path}")


def _tag_exists(tag: str) -> bool:
    completed = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/tags/{tag}"],
        cwd=PROJECT_ROOT, check=False,
    )
    return completed.returncode == 0


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True,
            text=True, timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise AnalysisContractError(f"Git freeze command failed: {args}") from exc


if __name__ == "__main__":
    main()

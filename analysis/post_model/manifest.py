from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from analysis.post_model import POST_MODEL_VERSION

from .common import (SEED, generated_file_hashes, sha256_file, utc_now, write_json)
from .verify import VerifiedInputs


def write_post_model_manifest(verified: VerifiedInputs, output: Path,
                              warnings: list[str]) -> dict[str, Any]:
    manifest_path = output / "post_model_manifest.json"
    excluded_files = {manifest_path}
    excluded_files.update(path for path in (output / "logs").rglob("*") if path.is_file())
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib",
                    "xgboost", "lightgbm", "catboost"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    try:
        post_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=verified.production_root.parents[1],
            check=True, capture_output=True, text=True, timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        post_commit = "unavailable"
    repository = verified.production_root.parents[1]
    try:
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
            check=True, capture_output=True, text=True, timeout=15,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        dirty = None
    source_files = sorted((repository / "analysis" / "post_model").glob("*.py"))
    payload = {
        "post_model_version": POST_MODEL_VERSION,
        "created_at_utc": utc_now(),
        "source_analysis_run_id": verified.manifest["analysis_run_id"],
        "source_dataset_version": verified.manifest["dataset_version"],
        "source_dataset_manifest_sha256": verified.contract["dataset"]["dataset_manifest_sha256"],
        "source_split_sha256": verified.manifest["split_hash"],
        "source_contract_version": verified.manifest["analysis_contract_version"],
        "source_analysis_commit": verified.manifest["analysis_git_commit"],
        "source_analysis_tag": verified.manifest["analysis_tag"],
        "post_model_code_commit": post_commit,
        "post_model_git_dirty": dirty,
        "post_model_source_hashes": {
            path.relative_to(repository).as_posix(): sha256_file(path) for path in source_files
        },
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "selected_model": verified.selection["selected_model"],
        "selected_model_sha256": sha256_file(
            verified.production_root / "modelling" / "models" / "selected_top4_f5.joblib"
        ),
        "seed": SEED,
        "warnings": warnings,
        "generated_files": generated_file_hashes(output, exclude=excluded_files),
        "production_artifact_hashes_before_and_after_equal": True,
    }
    write_json(manifest_path, payload)
    return payload

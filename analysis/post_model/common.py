from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from analysis.src.common import sha256_file, stable_hash, utc_now, write_json
from src.config import PROJECT_ROOT


TEXT_ARTIFACT_SUFFIXES = {".csv", ".json", ".md", ".toml", ".txt", ".yaml", ".yml"}


PRODUCTION_RUN_ID = "analysis-20260825T190121Z-1b3eaaff"
COLLECTION_RUN_ID = "20260823T124729Z-673b6e6c"
DATASET_VERSION = "tft_set17_patch17_8_v11"
CONTRACT_VERSION = "1.0.4"
ANALYSIS_TAG = "tft-patch17.8-analysis-contract-v5"
ANALYSIS_COMMIT = "dd1e0ed9bb8981536b5059d71324d5b7ad58a5cf"
SPLIT_HASH = "da64ca53fb584738b9db85c5364fd95644d399fe8202f9ad5b6619f271a95242"
SEED = 4275


def production_root(run_id: str = PRODUCTION_RUN_ID) -> Path:
    return PROJECT_ROOT / "outputs" / run_id


def default_output_root() -> Path:
    return PROJECT_ROOT / "artifacts" / "post_model"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_table(frame: pd.DataFrame, csv_path: Path, md_path: Path | None = None,
                decimals: int = 3) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    if md_path is not None:
        write_text(md_path, dataframe_markdown(frame, decimals=decimals))


def dataframe_markdown(frame: pd.DataFrame, decimals: int = 3) -> str:
    def display(value: Any) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{decimals}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(display(value) for value in row) + " |")
    return "\n".join(lines)


def safe_binary_metrics(y_true: Iterable[int], y_pred: Iterable[int],
                        probability: Iterable[float]) -> dict[str, float | None]:
    truth = np.asarray(list(y_true), dtype=int)
    predicted = np.asarray(list(y_pred), dtype=int)
    score = np.asarray(list(probability), dtype=float)
    result: dict[str, float | None] = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "roc_auc": None,
        "pr_auc": None,
    }
    if np.unique(truth).size == 2:
        # Metric implementations can overshoot the closed [0, 1] interval by a
        # few ulps for very small strata (for example 1.0000000000000002).
        result["roc_auc"] = float(np.clip(roc_auc_score(truth, score), 0.0, 1.0))
        result["pr_auc"] = float(np.clip(average_precision_score(truth, score), 0.0, 1.0))
    return result


def feature_group(name: str) -> str:
    if name.startswith(("unit_presence__", "unit_star__")):
        return "units_champions"
    if name.startswith(("trait_presence__", "trait_strength__")):
        return "traits"
    if name.startswith("item_count__"):
        return "items"
    if name.startswith("structure__"):
        return "composition_structure"
    if name in {"level", "composition_size"}:
        return "player_end_state_numeric"
    return "other_contract_approved"


def observation_id(participant_hash: str) -> str:
    return hashlib.sha256(("post-model:" + participant_hash).encode()).hexdigest()[:20]


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tft_post_model")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)sZ %(levelname)s %(message)s",
                                           datefmt="%Y-%m-%dT%H:%M:%S"))
    logger.addHandler(handler)
    return logger


def artifact_sha256_file(path: Path) -> str:
    """Hash generated text with canonical newlines and binary files byte-for-byte."""
    raw = path.read_bytes()
    if path.suffix.lower() in TEXT_ARTIFACT_SUFFIXES:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def generated_file_hashes(root: Path, exclude: set[Path] | None = None) -> list[dict[str, Any]]:
    excluded = {path.resolve() for path in (exclude or set())}
    return [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
         "sha256": artifact_sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.resolve() not in excluded
    ]


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, np.bool_):
        return bool(value)
    return value


__all__ = [
    "ANALYSIS_COMMIT", "ANALYSIS_TAG", "COLLECTION_RUN_ID", "CONTRACT_VERSION",
    "DATASET_VERSION", "PROJECT_ROOT", "PRODUCTION_RUN_ID", "SEED", "SPLIT_HASH",
    "artifact_sha256_file", "dataframe_markdown", "default_output_root", "feature_group",
    "generated_file_hashes", "json_sanitize", "observation_id", "production_root", "read_json",
    "safe_binary_metrics", "setup_logger", "sha256_file", "stable_hash", "utc_now",
    "write_json", "write_table", "write_text",
]

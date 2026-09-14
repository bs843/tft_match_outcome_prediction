from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.post_model.common import feature_group, safe_binary_metrics
from analysis.post_model.diagnostics import (confusion_summary, match_cluster_bootstrap,
                                             safe_misclassification_table)
from analysis.post_model.performance import calculate_f1_gaps, validation_ranking
from analysis.post_model.robustness import _stratified_metrics
from analysis.post_model.verify import validate_output_path, validate_split_frame


def test_split_invariants_and_overlap() -> None:
    frame = pd.DataFrame({
        "match_hash": ["a", "b", "c", "d"],
        "split": ["train", "train", "validation", "test"],
    })
    assert validate_split_frame(frame, {"train": 2, "validation": 1, "test": 1}) == []
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    failures = validate_split_frame(duplicated, {"train": 2, "validation": 1, "test": 1})
    assert any("overlap" in failure for failure in failures)


def test_output_path_never_overwrites_or_enters_production(tmp_path: Path) -> None:
    production = tmp_path / "production"
    production.mkdir()
    existing = tmp_path / "existing"
    existing.mkdir()
    assert validate_output_path(tmp_path / "new", production) == []
    assert any("already exists" in value for value in validate_output_path(existing, production))
    assert any("inside" in value for value in
               validate_output_path(production / "post", production, False))


def test_model_selection_ignores_test_metrics() -> None:
    frame = pd.DataFrame({
        "model": ["validation_winner", "test_winner"],
        "f1": [0.8, 0.7],
        "test_f1": [0.1, 0.99],
    })
    assert validation_ranking(frame).iloc[0]["model"] == "validation_winner"


def test_generalisation_gaps_are_signed_and_absolute() -> None:
    frame = pd.DataFrame({"model": ["m"], "cv_f1_mean": [0.80], "cv_f1_std": [0.02],
                          "validation_f1": [0.78], "test_f1": [0.75]})
    row = calculate_f1_gaps(frame).iloc[0]
    assert np.isclose(row["cv_to_validation_f1_gap"], -0.02)
    assert np.isclose(row["validation_to_test_f1_gap"], -0.03)
    assert np.isclose(row["cv_to_test_f1_gap"], -0.05)
    assert np.isclose(row["abs_cv_to_test_f1_gap"], 0.05)
    assert np.isclose(row["cv_to_test_gap_in_sd"], 2.5)


def test_confusion_matrix_calculations() -> None:
    result = confusion_summary([0, 0, 1, 1], [0, 1, 0, 1])
    assert result["true_negative"] == result["false_positive"] == 1
    assert result["false_negative"] == result["true_positive"] == 1
    assert result["specificity"] == result["sensitivity_recall"] == 0.5


def test_rank_and_region_stratification_and_single_class_auc() -> None:
    frame = pd.DataFrame({
        "match_hash": ["a", "a", "b", "b"], "top4": [0, 0, 0, 1],
        "prediction": [0, 0, 1, 1], "probability": [0.1, 0.2, 0.6, 0.8],
        "rank": ["GOLD", "GOLD", "MASTER", "MASTER"],
    })
    result = _stratified_metrics(frame, "rank").set_index("rank")
    assert result.loc["GOLD", "n"] == 2
    assert bool(result.loc["GOLD", "roc_auc_defined"]) is False
    assert pd.isna(result.loc["GOLD", "roc_auc"])
    assert bool(result.loc["MASTER", "roc_auc_defined"]) is True


def test_match_cluster_bootstrap_declares_match_resampling() -> None:
    rows = []
    for match in range(4):
        for slot in range(8):
            truth = int(slot < 4)
            rows.append({"match_hash": f"m{match}", "top4": truth,
                         "prediction": truth, "probability": 0.9 if truth else 0.1})
    result = match_cluster_bootstrap(pd.DataFrame(rows), samples=10, seed=4275)
    assert set(result["resampling_unit"]) == {"match"}
    assert set(result["bootstrap_samples"]) == {10}


def test_feature_group_mapping_is_deterministic() -> None:
    expected = {
        "unit_presence__aatrox": "units_champions",
        "unit_star__aatrox": "units_champions",
        "trait_strength__anima": "traits",
        "item_count__rageblade": "items",
        "structure__active_traits": "composition_structure",
        "level": "player_end_state_numeric",
    }
    assert {name: feature_group(name) for name in expected} == expected


def test_no_prohibited_identifiers_in_misclassification_output() -> None:
    frame = pd.DataFrame([{
        "participant_hash": "private-hash", "top4": 1, "prediction": 0,
        "probability": 0.1, "placement": 4, "rank_tier": "GOLD",
        "rank_division": "I", "rank_status": "found", "platform_region": "EUW1",
        "regional_route": "EUROPE",
    }])
    safe = safe_misclassification_table(frame)
    assert "participant_hash" not in safe.columns
    assert "match_hash" not in safe.columns
    assert safe.loc[0, "observation_id"] != "private-hash"


def test_safe_metrics_returns_undefined_auc_for_single_class() -> None:
    metrics = safe_binary_metrics([0, 0], [0, 0], [0.1, 0.2])
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None


def test_safe_metrics_clamps_floating_point_auc_overshoot(monkeypatch) -> None:
    monkeypatch.setattr("analysis.post_model.common.roc_auc_score",
                        lambda truth, score: 1.0 + np.finfo(float).eps)
    monkeypatch.setattr("analysis.post_model.common.average_precision_score",
                        lambda truth, score: 1.0 + np.finfo(float).eps)
    metrics = safe_binary_metrics([0, 1], [0, 1], [0.1, 0.9])
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0

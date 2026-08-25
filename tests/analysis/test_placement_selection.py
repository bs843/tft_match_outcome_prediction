from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

import analysis.src.models as model_module
from analysis.src.common import stable_hash, write_json
from analysis.src.contract import load_contract
from analysis.src.models import (PLACEMENT_TASK, SELECTION_MANIFESTS, TOP4_TASK,
                                 _selection_manifest, approved_model_registry,
                                 authorize_final_test_access, run_development_selection,
                                 run_final_evaluation, verify_locked_selection)


def _frame(matches: int, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    features = []
    for match in range(matches):
        for placement in range(1, 9):
            rows.append({
                "match_hash": f"{prefix}-{match}",
                "participant_hash": f"{prefix}-{match}-{placement}",
                "top4": int(placement <= 4),
                "placement": placement,
            })
            features.append({"placement_signal": float(placement),
                             "stable_noise": float(match % 2)})
    return pd.DataFrame(rows), pd.DataFrame(features)


def _context(run_id: str = "analysis-test") -> dict[str, str]:
    return {
        "analysis_run_id": run_id,
        "contract_version": "1.0.4",
        "analysis_tag": "tft-patch17.8-analysis-contract-v5",
        "git_commit": "test-commit",
        "dataset_version": "tft_set17_patch17_8_v11",
        "split_hash": "da64ca53fb584738b9db85c5364fd95644d399fe8202f9ad5b6619f271a95242",
        "preprocessing_sha256": "preprocessing-hash",
    }


def _lightweight_top4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_module, "_estimators",
        lambda seed, require_boosters: {
            "dummy": DummyClassifier(strategy="prior", random_state=seed)
        },
    )


def _write_selections(output: Path) -> dict[str, dict[str, Any]]:
    registry = approved_model_registry()
    context = _context()
    feature_hash = stable_hash({"family": "F5", "columns": ["x"]})
    top4 = _selection_manifest(
        context, task=TOP4_TASK, target_definition="top4 = placement <= 4",
        selected_model="dummy", model_contract=registry["primary_registry"]["dummy"],
        validation_metrics={"f1": 0.5}, feature_hash=feature_hash, seed=4275,
        candidate_validation_results={"dummy": {"f1": 0.5}},
        metric="f1", direction="maximize", tie_break="primary_registry_order",
        registry_order=["dummy"], threshold=0.5,
    )
    placement = _selection_manifest(
        context, task=PLACEMENT_TASK, target_definition="validated final placement 1-8",
        selected_model="dummy",
        model_contract=registry["secondary_placement_registry"]["dummy"],
        validation_metrics={"mae": 2.0, "accuracy": 0.125,
                            "macro_f1": 0.03, "weighted_f1": 0.03},
        candidate_validation_results={
            "dummy": {"mae": 2.0}, "random_forest": {"mae": 1.5}
        },
        feature_hash=feature_hash, seed=4275, metric="mae", direction="minimize",
        tie_break="secondary_placement_registry_order",
        registry_order=["dummy", "random_forest"], threshold=None,
    )
    output.mkdir(parents=True)
    write_json(output / SELECTION_MANIFESTS[TOP4_TASK], top4)
    write_json(output / SELECTION_MANIFESTS[PLACEMENT_TASK], placement)
    return {TOP4_TASK: top4, PLACEMENT_TASK: placement}


def test_exact_placement_candidates_receive_validation_mae_and_lowest_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _lightweight_top4(monkeypatch)
    train, train_x = _frame(6, "train")
    validation, validation_x = _frame(2, "validation")
    selections = run_development_selection(
        train, validation, {"F5": (train_x, validation_x)}, tmp_path / "modelling",
        seed=4275, folds=2, context=_context(),
    )
    rows = pd.read_csv(tmp_path / "modelling" / "tables" / "validation_metrics.csv")
    placement = rows[rows["task"] == PLACEMENT_TASK]
    assert set(placement["model"]) == {"dummy", "random_forest"}
    assert placement["mae"].notna().all()
    assert placement["config_identifier"].str.len().eq(64).all()
    assert placement["feature_identity_sha256"].str.len().eq(64).all()
    assert placement["analysis_run_id"].eq("analysis-test").all()
    winner = placement.loc[placement["mae"].idxmin(), "model"]
    assert selections[PLACEMENT_TASK]["selected_model"] == winner
    assert placement.loc[placement["model"] == winner, "selected"].all()
    assert selections[PLACEMENT_TASK]["validation_metrics"]["mae"] == pytest.approx(
        placement["mae"].min()
    )
    assert selections[PLACEMENT_TASK]["task"] == PLACEMENT_TASK
    assert selections[TOP4_TASK]["task"] == TOP4_TASK
    assert "test" not in inspect.signature(run_development_selection).parameters


def test_exact_placement_tie_is_deterministic_and_prefers_registry_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _lightweight_top4(monkeypatch)
    monkeypatch.setattr(
        model_module, "_placement_estimators",
        lambda seed: {
            "dummy": DummyClassifier(strategy="most_frequent", random_state=seed),
            "random_forest": DummyClassifier(strategy="most_frequent", random_state=seed),
        },
    )
    train, train_x = _frame(4, "train")
    validation, validation_x = _frame(2, "validation")
    winners = []
    for index in range(2):
        selections = run_development_selection(
            train, validation, {"F5": (train_x, validation_x)},
            tmp_path / f"modelling-{index}", seed=4275, folds=2,
            context=_context(f"analysis-test-{index}"),
        )
        winners.append(selections[PLACEMENT_TASK]["selected_model"])
    assert winners == ["dummy", "dummy"]


def test_selection_manifest_integrity_and_cross_task_isolation(tmp_path: Path) -> None:
    output = tmp_path / "modelling"
    selections = _write_selections(output)
    verify_locked_selection(
        output / SELECTION_MANIFESTS[PLACEMENT_TASK], expected_task=PLACEMENT_TASK,
        expected_selected_model="dummy",
    )
    with pytest.raises(RuntimeError, match="cannot authorize"):
        verify_locked_selection(
            output / SELECTION_MANIFESTS[TOP4_TASK], expected_task=PLACEMENT_TASK,
        )
    with pytest.raises(RuntimeError, match="cannot authorize"):
        verify_locked_selection(
            output / SELECTION_MANIFESTS[PLACEMENT_TASK], expected_task=TOP4_TASK,
        )
    tampered = dict(selections[PLACEMENT_TASK])
    tampered["selected_model"] = "random_forest"
    write_json(output / SELECTION_MANIFESTS[PLACEMENT_TASK], tampered)
    with pytest.raises(RuntimeError, match="integrity failed"):
        verify_locked_selection(
            output / SELECTION_MANIFESTS[PLACEMENT_TASK], expected_task=PLACEMENT_TASK,
        )


def test_access_is_recorded_before_payload_and_second_access_is_rejected(
    tmp_path: Path,
) -> None:
    output = tmp_path / "modelling"
    selections = _write_selections(output)
    manifest_path = tmp_path / "run_manifest.json"
    run_manifest: dict[str, Any] = {"test_accessed": False, "test_access_by_task": {}}
    events = authorize_final_test_access(output, selections, run_manifest, manifest_path)
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "final_evaluation_authorized"
    assert persisted["test_accessed"] is True
    assert set(events) == {TOP4_TASK, PLACEMENT_TASK}
    assert all(event["access_count"] == 1 for event in events.values())
    with pytest.raises(RuntimeError, match="already been accessed"):
        authorize_final_test_access(output, selections, run_manifest, manifest_path)


def test_final_evaluation_rejects_missing_tampered_or_mismatched_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _lightweight_top4(monkeypatch)
    output = tmp_path / "modelling"
    selections = _write_selections(output)
    (output / SELECTION_MANIFESTS[PLACEMENT_TASK]).unlink()
    with pytest.raises(RuntimeError, match="missing"):
        run_final_evaluation(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}, {}, output,
            selections=selections, seed=4275, access_events={},
        )
    write_json(output / SELECTION_MANIFESTS[PLACEMENT_TASK], selections[PLACEMENT_TASK])
    mismatched = {**selections, PLACEMENT_TASK: {
        **selections[PLACEMENT_TASK], "selected_model": "random_forest"
    }}
    with pytest.raises(RuntimeError, match="differs from locked manifest"):
        run_final_evaluation(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}, {}, output,
            selections=mismatched, seed=4275, access_events={},
        )


def test_only_selected_placement_model_reaches_final_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _lightweight_top4(monkeypatch)
    monkeypatch.setattr(model_module.joblib, "dump", lambda *_args, **_kwargs: None)
    calls: list[str] = []

    class SpyPlacement:
        def __init__(self, name: str) -> None:
            self.name = name

        def fit(self, _x, _y):
            calls.append(self.name)
            return self

        def predict(self, x):
            return np.ones(len(x), dtype=int)

    monkeypatch.setattr(
        model_module, "_placement_estimators",
        lambda seed: {"dummy": SpyPlacement("dummy"),
                      "random_forest": SpyPlacement("random_forest")},
    )
    output = tmp_path / "modelling"
    selections = _write_selections(output)
    run_manifest: dict[str, Any] = {"test_accessed": False, "test_access_by_task": {}}
    events = authorize_final_test_access(
        output, selections, run_manifest, tmp_path / "run_manifest.json"
    )
    base, matrix = _frame(1, "match")
    result = run_final_evaluation(
        base, base, base, {"F5": matrix}, {"F5": matrix}, {"F5": matrix}, output,
        selections=selections, seed=4275, access_events=events,
    )
    final_rows = pd.read_csv(output / "tables" / "final_test_metrics.csv")
    assert result["test_evaluation_count"] == 1
    assert calls == ["dummy"]
    assert final_rows.loc[final_rows["target"] == "placement", "model"].tolist() == ["dummy"]


def test_v104_preserves_frozen_science_without_historical_git_tags() -> None:
    current = load_contract()
    current.pop("_path", None)
    current.pop("analysis_contract_version", None)
    current.pop("release", None)
    current.pop("amendment", None)
    current["selection"].pop("exact_placement", None)
    for key in (
        "task_aware_locked_manifests", "independently_locked_tasks",
        "exact_placement_requires_locked_selection",
        "exact_placement_manifest_integrity_required",
        "direct_multi_model_placement_test_evaluation",
    ):
        current["test_access"].pop(key, None)
    current["production_artifacts"].pop("required", None)
    assert stable_hash(current) == (
        "666a38ce411958a647cbcf06ad5f4b08e928defe50b464547b3f02cd023f53e0"
    )
    frozen = load_contract()
    assert frozen["split"]["counts"] == {"train": 3500, "validation": 750, "test": 750}
    assert frozen["split"]["canonical_assignment_sha256"] == (
        "da64ca53fb584738b9db85c5364fd95644d399fe8202f9ad5b6619f271a95242"
    )
    assert frozen["seeds"]["master_seed"] == 4275
    assert list(frozen["models"]["secondary_placement_registry"]) == [
        "dummy", "random_forest"
    ]

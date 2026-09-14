"""Non-UI safety and provenance tests for the read-only dashboard."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from dashboard.components.loaders import (
    ArtifactCatalog,
    load_analysis_contract,
    load_class_distribution,
    load_composition_manifest,
    load_composition_table,
    load_dataset_summary,
    load_exact_placement_metrics,
    load_evidence_synthesis_manifest,
    load_evidence_synthesis_table,
    load_feature_family_metrics,
    load_feature_importance,
    load_final_test_metrics,
    load_metric_confidence_intervals,
    load_rank_stratified_metrics,
    load_validation_metrics,
    verify_post_model_bundle,
)
from dashboard.components.validation import (
    MetricScopeError,
    ProhibitedColumnsError,
    assert_metric_scope,
    assert_safe_columns,
)


ROOT = Path(__file__).resolve().parents[2]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_artifact_paths_resolve() -> None:
    paths = ArtifactCatalog(ROOT)
    assert paths.contract.is_file()
    assert paths.verification_report.is_file()
    assert paths.post_model_manifest.is_file()
    assert paths.production_root().is_dir()
    assert (paths.extension / "tables" / "placement_final_test_metrics.csv").is_file()


def test_metric_loaders_return_expected_columns() -> None:
    assert {"model", "f1", "accuracy", "partition"} <= set(load_validation_metrics(ROOT))
    assert {"target", "model", "f1", "partition"} <= set(load_final_test_metrics(ROOT))
    assert {"feature_family", "f1", "model", "partition"} <= set(
        load_feature_family_metrics(ROOT)
    )


def test_validation_and_final_test_scopes_cannot_be_confused() -> None:
    assert set(load_validation_metrics(ROOT)["partition"]) == {"validation"}
    assert set(load_final_test_metrics(ROOT)["partition"]) == {"test"}
    with pytest.raises(MetricScopeError):
        assert_metric_scope(load_final_test_metrics(ROOT), "validation")


@pytest.mark.parametrize(
    "column",
    ["puuid", "match_hash", "participant_id", "riot_id_value", "api_key"],
)
def test_prohibited_identifier_columns_are_rejected(column: str) -> None:
    with pytest.raises(ProhibitedColumnsError):
        assert_safe_columns(["metric", column])


def test_dashboard_loaders_do_not_write_frozen_sources() -> None:
    observed = [
        ROOT / "config" / "analysis_contract.yaml",
        ROOT / "artifacts" / "post_model" / "tables" / "dataset_summary.csv",
        ROOT / "artifacts" / "post_model" / "tables" / "final_test_model_summary.csv",
        ROOT
        / "artifacts"
        / "thesis_extension_v1"
        / "tables"
        / "feature_family_validation_comparison.csv",
    ]
    before = {path: (_digest(path), path.stat().st_mtime_ns) for path in observed}
    load_analysis_contract(ROOT)
    load_dataset_summary(ROOT)
    load_final_test_metrics(ROOT)
    load_feature_family_metrics(ROOT)
    load_feature_importance(ROOT)
    load_exact_placement_metrics(ROOT)
    after = {path: (_digest(path), path.stat().st_mtime_ns) for path in observed}
    assert before == after


def test_dashboard_imports_no_training_or_model_libraries() -> None:
    forbidden_modules = {
        "analysis",
        "catboost",
        "joblib",
        "lightgbm",
        "sklearn",
        "xgboost",
    }
    forbidden_calls = {
        "cross_validate",
        "fit",
        "fit_predict",
        "fit_transform",
        "grid_search",
        "predict",
        "predict_proba",
        "randomized_search",
        "train",
    }
    for path in (ROOT / "dashboard").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not {name.name.split(".")[0] for name in node.names} & forbidden_modules
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden_modules
            elif isinstance(node, ast.Call):
                function = node.func
                name = function.attr if isinstance(function, ast.Attribute) else (
                    function.id if isinstance(function, ast.Name) else ""
                )
                assert name.lower() not in forbidden_calls


def test_expected_split_total_equals_five_thousand_matches() -> None:
    counts = load_analysis_contract(ROOT)["split"]["counts"]
    assert sum(int(value) for value in counts.values()) == 5000


def test_expected_participant_total_equals_forty_thousand() -> None:
    summary = dict(
        zip(
            load_dataset_summary(ROOT)["measure"],
            load_dataset_summary(ROOT)["value"],
            strict=False,
        )
    )
    assert int(float(summary["accepted_matches"])) == 5000
    assert int(float(summary["participant_rows"])) == 40000
    distribution = load_class_distribution(ROOT)
    assert int(distribution["participants"].sum()) == 40000


def test_displayed_primary_metrics_match_frozen_sources_exactly() -> None:
    loaded = load_final_test_metrics(ROOT)
    source = pd.read_csv(
        ROOT / "artifacts" / "post_model" / "tables" / "final_test_model_summary.csv"
    )
    loaded_row = loaded.loc[loaded["target"] == "top4"].iloc[0]
    source_row = source.loc[source["target"] == "top4"].iloc[0]
    for metric in ("f1", "accuracy", "roc_auc", "pr_auc"):
        assert loaded_row[metric] == source_row[metric]


def test_displayed_f1_interval_matches_frozen_source() -> None:
    loaded = load_metric_confidence_intervals(ROOT).set_index("metric").loc["f1"]
    source = pd.read_csv(
        ROOT / "artifacts" / "post_model" / "tables" / "final_test_metric_ci.csv"
    ).set_index("metric").loc["f1"]
    assert loaded["lower_95"] == source["lower_95"]
    assert loaded["upper_95"] == source["upper_95"]


def test_missing_optional_rank_artifact_fails_gracefully(tmp_path: Path) -> None:
    assert load_rank_stratified_metrics(tmp_path) is None


def test_post_model_manifest_verifies() -> None:
    result = verify_post_model_bundle(ROOT)
    assert result["status"] == "PASS", result["failures"]
    assert result["verified"] == result["expected"]


def test_composition_dashboard_loaders_are_read_only_and_identifier_safe() -> None:
    if not (ROOT / "artifacts" / "post_model" / "composition_analysis_v1").is_dir():
        return
    manifest = load_composition_manifest(ROOT)
    assert manifest["status"] == "PASS"
    units = load_composition_table("unit_top4_analysis.csv", ROOT)
    assert {"unit", "appearances", "top_four_rate", "top_four_uplift_pp"} <= set(units)
    assert_safe_columns(units.columns)


def test_evidence_synthesis_dashboard_loaders_are_allow_listed_and_safe() -> None:
    output = ROOT / "artifacts" / "post_model" / "evidence_synthesis"
    if not output.is_dir():
        return
    manifest = load_evidence_synthesis_manifest(ROOT)
    assert manifest["status"] == "PASS"
    evidence = load_evidence_synthesis_table("feature_evidence_table.csv", ROOT)
    assert {"feature", "feature_type", "evidence_category"} <= set(evidence)
    assert_safe_columns(evidence.columns)
    with pytest.raises(Exception):
        load_evidence_synthesis_table("unapproved.csv", ROOT)

"""Central read-only loaders for frozen modelling and post-model artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd
import yaml

from dashboard.components.validation import (
    MissingArtifactError,
    assert_metric_scope,
    assert_safe_columns,
    optional_file,
    require_file,
    verify_generated_manifest,
)

try:  # Allows non-UI loader tests to run before the optional dashboard extra is installed.
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - exercised only without dashboard extras.
    F = TypeVar("F", bound=Callable[..., Any])

    class _StreamlitCacheShim:
        @staticmethod
        def cache_data(*args: Any, **kwargs: Any) -> Callable[[F], F] | F:
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]

            def decorator(function: F) -> F:
                return function

            return decorator

    st = _StreamlitCacheShim()  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ArtifactCatalog:
    """Repository-relative map of authoritative frozen dashboard sources."""

    root: Path = REPO_ROOT

    @property
    def contract(self) -> Path:
        return self.root / "config" / "analysis_contract.yaml"

    @property
    def post_model(self) -> Path:
        return self.root / "artifacts" / "post_model"

    @property
    def extension(self) -> Path:
        return self.root / "artifacts" / "thesis_extension_v1"

    @property
    def composition_analysis(self) -> Path:
        return self.post_model / "composition_analysis_v1"

    @property
    def evidence_synthesis(self) -> Path:
        return self.post_model / "evidence_synthesis"

    @property
    def verification_report(self) -> Path:
        return self.post_model / "verification_report.json"

    @property
    def post_model_manifest(self) -> Path:
        return self.post_model / "post_model_manifest.json"

    def production_root(self) -> Path:
        report = _read_json(self.verification_report)
        return self.root / "outputs" / str(report["production_run_id"])


def catalog(root: Path | str | None = None) -> ArtifactCatalog:
    return ArtifactCatalog(Path(root).resolve() if root is not None else REPO_ROOT)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def _read_csv(path: Path, required: set[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(require_file(path))
    if required:
        missing = required.difference(frame.columns)
        if missing:
            raise MissingArtifactError(
                f"Frozen table {path} is missing columns: {sorted(missing)}"
            )
    assert_safe_columns(frame.columns)
    return frame


@st.cache_data(show_spinner=False)
def load_analysis_contract(root: Path | str | None = None) -> dict[str, Any]:
    path = catalog(root).contract
    return yaml.safe_load(require_file(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_verification_report(root: Path | str | None = None) -> dict[str, Any]:
    return _read_json(catalog(root).verification_report)


@st.cache_data(show_spinner=False)
def load_post_model_manifest(root: Path | str | None = None) -> dict[str, Any]:
    return _read_json(catalog(root).post_model_manifest)


@st.cache_data(show_spinner=False)
def load_dashboard_manifest(root: Path | str | None = None) -> dict[str, Any]:
    return _read_json(catalog(root).root / "dashboard" / "dashboard_manifest.json")


@st.cache_data(show_spinner=False)
def load_production_manifest(root: Path | str | None = None) -> dict[str, Any]:
    paths = catalog(root)
    return _read_json(paths.production_root() / "run_manifest.json")


@st.cache_data(show_spinner=False)
def load_selection_manifest(root: Path | str | None = None) -> dict[str, Any]:
    paths = catalog(root)
    return _read_json(paths.production_root() / "modelling" / "selection_manifest.json")


@st.cache_data(show_spinner=False)
def load_placement_selection_manifest(root: Path | str | None = None) -> dict[str, Any]:
    paths = catalog(root)
    return _read_json(
        paths.production_root() / "modelling" / "placement_selection_manifest.json"
    )


@st.cache_data(show_spinner=False)
def load_results_summary(root: Path | str | None = None) -> str:
    path = catalog(root).post_model / "thesis_results_summary.md"
    return require_file(path).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_dataset_summary(root: Path | str | None = None) -> pd.DataFrame:
    path = catalog(root).post_model / "tables" / "dataset_summary.csv"
    return _read_csv(path, {"measure", "value"})


@st.cache_data(show_spinner=False)
def load_final_test_metrics(root: Path | str | None = None) -> pd.DataFrame:
    path = catalog(root).post_model / "tables" / "final_test_model_summary.csv"
    frame = _read_csv(path, {"target", "model", "feature_family", "partition"})
    assert_metric_scope(frame, "test")
    frame = frame.copy()
    frame["metric_scope"] = "frozen_final_test"
    return frame


@st.cache_data(show_spinner=False)
def load_validation_metrics(root: Path | str | None = None) -> pd.DataFrame:
    path = catalog(root).post_model / "tables" / "validation_model_summary.csv"
    frame = _read_csv(path, {"model", "f1", "accuracy", "roc_auc", "selected"})
    frame = frame.copy()
    frame["partition"] = "validation"
    frame["metric_scope"] = "model_selection"
    assert_metric_scope(frame, "validation")
    return frame


@st.cache_data(show_spinner=False)
def load_cross_validation_metrics(root: Path | str | None = None) -> pd.DataFrame:
    path = catalog(root).post_model / "tables" / "cv_model_summary.csv"
    frame = _read_csv(path, {"model", "cv_f1_mean", "cv_f1_std", "cv_folds"})
    frame = frame.copy()
    frame["partition"] = "development_cross_validation"
    return frame


@st.cache_data(show_spinner=False)
def load_feature_family_metrics(root: Path | str | None = None) -> pd.DataFrame:
    path = (
        catalog(root).extension
        / "tables"
        / "feature_family_validation_comparison.csv"
    )
    frame = _read_csv(
        path,
        {"feature_family", "accuracy", "f1", "roc_auc", "pr_auc"},
    ).copy()
    frame["model"] = "logistic_feature_family"
    frame["partition"] = "validation"
    frame["metric_scope"] = "prespecified_feature_family_comparison"
    assert_metric_scope(frame, "validation")
    return frame


@st.cache_data(show_spinner=False)
def load_feature_family_definitions(root: Path | str | None = None) -> pd.DataFrame:
    path = catalog(root).extension / "tables" / "feature_family_definitions.csv"
    return _read_csv(
        path,
        {"feature_family", "cumulative", "included_units", "variables_or_patterns"},
    )


@st.cache_data(show_spinner=False)
def load_metric_confidence_intervals(root: Path | str | None = None) -> pd.DataFrame:
    path = catalog(root).post_model / "tables" / "final_test_metric_ci.csv"
    return _read_csv(
        path,
        {"metric", "point_estimate", "lower_95", "upper_95", "resampling_unit"},
    )


@st.cache_data(show_spinner=False)
def load_confusion_matrix(root: Path | str | None = None) -> pd.DataFrame:
    path = catalog(root).post_model / "tables" / "final_test_confusion_matrix.csv"
    frame = pd.read_csv(require_file(path), index_col=0)
    assert_safe_columns(frame.columns)
    return frame


@st.cache_data(show_spinner=False)
def load_roc_points(root: Path | str | None = None) -> pd.DataFrame:
    path = catalog(root).post_model / "tables" / "final_test_roc_points.csv"
    return _read_csv(path, {"false_positive_rate", "true_positive_rate"})


@st.cache_data(show_spinner=False)
def load_pr_points(root: Path | str | None = None) -> pd.DataFrame:
    path = catalog(root).post_model / "tables" / "final_test_pr_points.csv"
    return _read_csv(path, {"recall", "precision"})


@st.cache_data(show_spinner=False)
def load_feature_importance(root: Path | str | None = None) -> pd.DataFrame:
    paths = catalog(root)
    shap = paths.extension / "tables" / "shap_feature_importance.csv"
    if optional_file(shap):
        frame = _read_csv(
            shap,
            {"feature", "mean_absolute_shap", "substantive_unit"},
        ).copy()
        frame["importance_method"] = "TreeSHAP (frozen validation analysis)"
        return frame.sort_values("mean_absolute_shap", ascending=False)

    fallback = paths.post_model / "tables" / "native_xgboost_feature_importance.csv"
    frame = _read_csv(fallback, {"feature", "importance"}).copy()
    frame["importance_method"] = "XGBoost native feature importance"
    return frame.sort_values("importance", ascending=False)


@st.cache_data(show_spinner=False)
def load_feature_group_importance(root: Path | str | None = None) -> pd.DataFrame:
    path = catalog(root).post_model / "tables" / "feature_group_importance.csv"
    return _read_csv(
        path,
        {"feature_group", "feature_count", "absolute_mean_importance_sum"},
    )


@st.cache_data(show_spinner=False)
def load_class_distribution(root: Path | str | None = None) -> pd.DataFrame:
    path = catalog(root).post_model / "eda" / "class_distribution.csv"
    return _read_csv(path, {"outcome", "participants", "proportion"})


@st.cache_data(show_spinner=False)
def load_placement_distribution(root: Path | str | None = None) -> pd.DataFrame:
    path = catalog(root).post_model / "eda" / "placement_distribution.csv"
    return _read_csv(path, {"placement", "participants", "proportion"})


@st.cache_data(show_spinner=False)
def load_outcome_summary(root: Path | str | None = None) -> pd.DataFrame:
    path = catalog(root).post_model / "eda" / "composition_structure_by_outcome.csv"
    return _read_csv(path, {"outcome", "variable", "n", "mean", "median", "q1", "q3"})


@st.cache_data(show_spinner=False)
def load_exact_placement_metrics(root: Path | str | None = None) -> pd.DataFrame:
    path = catalog(root).extension / "tables" / "placement_final_test_metrics.csv"
    frame = _read_csv(path, {"model", "partition", "accuracy", "macro_f1", "mae"})
    assert_metric_scope(frame, "final_test_read_only")
    return frame


@st.cache_data(show_spinner=False)
def load_exact_placement_ci(root: Path | str | None = None) -> pd.DataFrame:
    path = catalog(root).extension / "tables" / "placement_match_bootstrap_ci.csv"
    return _read_csv(path, {"metric", "estimate", "ci95_low", "ci95_high"})


@st.cache_data(show_spinner=False)
def load_exact_placement_confusion(root: Path | str | None = None) -> pd.DataFrame:
    path = catalog(root).extension / "tables" / "placement_final_test_confusion_matrix.csv"
    frame = pd.read_csv(require_file(path), index_col=0)
    assert_safe_columns(frame.columns)
    return frame


@st.cache_data(show_spinner=False)
def load_rank_stratified_metrics(root: Path | str | None = None) -> pd.DataFrame | None:
    path = catalog(root).post_model / "tables" / "test_metrics_by_rank.csv"
    if optional_file(path) is None:
        return None
    return _read_csv(path, {"rank_stratum", "n", "matches", "f1"})


COMPOSITION_TABLES = {
    "adjusted_logistic_regression.csv",
    "feature_family_interpretation.csv",
    "feature_interactions.csv",
    "global_baseline.csv",
    "item_top4_analysis.csv",
    "rank_stratified_feature_analysis.csv",
    "shap_feature_family_importance.csv",
    "shap_feature_importance.csv",
    "trait_threshold_comparisons.csv",
    "trait_top4_analysis.csv",
    "unit_item_holder_analysis.csv",
    "unit_star_top4_analysis.csv",
    "unit_top4_analysis.csv",
}


@st.cache_data(show_spinner=False)
def load_composition_table(name: str, root: Path | str | None = None) -> pd.DataFrame:
    """Load one allow-listed, identifier-free composition-analysis table."""
    if name not in COMPOSITION_TABLES:
        raise MissingArtifactError(f"Composition table is not approved for display: {name}")
    return _read_csv(catalog(root).composition_analysis / "tables" / name)


@st.cache_data(show_spinner=False)
def load_composition_manifest(root: Path | str | None = None) -> dict[str, Any]:
    return _read_json(catalog(root).composition_analysis / "composition_analysis_manifest.json")


EVIDENCE_SYNTHESIS_TABLES = {
    "feature_evidence_table.csv",
    "feature_family_incremental_value.csv",
    "interaction_strength_comparison.csv",
    "multiple_testing_summary.csv",
    "raw_vs_adjusted_associations.csv",
    "shap_vs_descriptive.csv",
    "support_threshold_sensitivity.csv",
    "threshold_rank_stability.csv",
    "train_validation_descriptive_stability.csv",
}


@st.cache_data(show_spinner=False)
def load_evidence_synthesis_table(name: str,
                                  root: Path | str | None = None) -> pd.DataFrame:
    """Load one allow-listed, identifier-free synthesis table."""
    if name not in EVIDENCE_SYNTHESIS_TABLES:
        raise MissingArtifactError(f"Evidence-synthesis table is not approved: {name}")
    return _read_csv(catalog(root).evidence_synthesis / name)


@st.cache_data(show_spinner=False)
def load_evidence_synthesis_manifest(root: Path | str | None = None) -> dict[str, Any]:
    return _read_json(catalog(root).evidence_synthesis / "evidence_synthesis_manifest.json")


@st.cache_data(show_spinner=False)
def load_limitations_matrix(root: Path | str | None = None) -> str:
    return require_file(catalog(root).post_model / "limitations_matrix.md").read_text(
        encoding="utf-8"
    )


@st.cache_data(show_spinner=False)
def verify_post_model_bundle(root: Path | str | None = None) -> dict[str, object]:
    paths = catalog(root)
    return verify_generated_manifest(paths.post_model_manifest, paths.post_model)


def frozen_figure(relative_path: str, root: Path | str | None = None) -> Path | None:
    """Return an optional frozen image under the approved artifact roots only."""
    paths = catalog(root)
    candidate = (paths.root / relative_path).resolve()
    approved_roots = (
        paths.post_model.resolve(),
        paths.extension.resolve(),
        paths.composition_analysis.resolve(),
        paths.evidence_synthesis.resolve(),
    )
    if not any(candidate == base or base in candidate.parents for base in approved_roots):
        return None
    return optional_file(candidate)

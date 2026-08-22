from __future__ import annotations

from analysis.src.models import _estimators


def test_primary_baseline_registry_includes_all_boosters() -> None:
    estimators = _estimators(seed=4275, require_boosters=False)
    assert set(estimators) == {
        "dummy", "logistic_regression", "random_forest", "xgboost",
        "lightgbm", "catboost",
    }
    assert estimators["dummy"] is not None
    assert estimators["logistic_regression"] is not None
    assert estimators["random_forest"] is not None

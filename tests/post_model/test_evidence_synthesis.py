from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.composition_analysis.run import locate_production
from src.post_model.evidence_synthesis import (
    PRIMARY_THRESHOLDS,
    descriptive_evidence,
    feature_evidence_table,
    interaction_strength,
    threshold_sensitivity,
)


ROOT = Path(__file__).resolve().parents[2]
COMPOSITION = ROOT / "artifacts" / "post_model" / "composition_analysis_v1"
OUTPUT = ROOT / "artifacts" / "post_model" / "evidence_synthesis"


def _row(frame: pd.DataFrame, feature_type: str, feature: str) -> pd.Series:
    return frame.loc[(frame.feature_type == feature_type) & (frame.feature == feature)].iloc[0]


def test_headline_descriptive_findings_match_authoritative_artifacts() -> None:
    frame = descriptive_evidence()
    shen = _row(frame, "unit", "Shen")
    aurora = _row(frame, "unit", "Aurora")
    force = _row(frame, "item", "Force Of Nature")
    space = _row(frame, "trait_signature", "Space Groove · tier 4 · 7 units")
    assert np.isclose(shen.top4_rate, .658, atol=.001)
    assert np.isclose(shen.uplift_pp, 15.8, atol=.1)
    assert np.isclose(aurora.top4_rate, .388, atol=.001)
    assert np.isclose(aurora.uplift_pp, -11.2, atol=.1)
    assert np.isclose(force.top4_rate, .770, atol=.001)
    assert np.isclose(force.uplift_pp, 27.0, atol=.1)
    assert np.isclose(space.top4_rate, .708, atol=.001)
    assert np.isclose(space.uplift_pp, 20.8, atol=.1)


def test_headline_interaction_matches_authoritative_artifact() -> None:
    frame = interaction_strength()
    row = frame.loc[
        ((frame.feature_a == "Hptank · tier 1 · 2 units") &
         (frame.feature_b == "Shen Unique · tier 1 · 1 units")) |
        ((frame.feature_b == "Hptank · tier 1 · 2 units") &
         (frame.feature_a == "Shen Unique · tier 1 · 1 units"))
    ].iloc[0]
    assert row.appearances == 2749
    assert np.isclose(row.top_four_rate, .708, atol=.001)
    assert "not synergy" in row.interpretation


def test_evidence_mapping_is_explicit_and_rank_remains_excluded() -> None:
    frame = feature_evidence_table()
    shen = _row(frame, "unit", "Shen")
    assert shen.adjusted_odds_ratio > 1
    assert shen.mean_abs_shap > 0
    assert shen.direction_consistent
    assert not frame.feature_type.eq("rank_context").any()


def test_support_sensitivity_uses_only_predeclared_thresholds() -> None:
    raw = descriptive_evidence()
    interactions = interaction_strength()
    sensitivity, _ = threshold_sensitivity(raw, interactions)
    for feature_type, values in sensitivity.groupby("feature_type")["support_threshold"]:
        assert feature_type in PRIMARY_THRESHOLDS
        assert set(values).issubset({100, 150, 200, 250, 300, 500})
    assert 100 not in set(sensitivity.loc[sensitivity.feature_type == "interaction",
                                         "support_threshold"])


def test_frozen_cohort_has_eight_rows_and_four_top_four_outcomes_per_match() -> None:
    try:
        production = locate_production()
    except RuntimeError:
        pytest.skip("Private frozen cohort is unavailable in this source checkout")
    frames = [pd.read_csv(production / "data" / f"{split}_participants.csv.gz",
                          usecols=["match_hash", "placement", "top4"])
              for split in ("train", "validation", "test")]
    cohort = pd.concat(frames, ignore_index=True)
    grouped = cohort.groupby("match_hash")
    assert len(cohort) == 40_000
    assert grouped.ngroups == 5_000
    assert grouped.size().eq(8).all()
    assert grouped.top4.sum().eq(4).all()
    assert cohort.top4.mean() == .5
    assert cohort.placement.mean() == 4.5


def test_generated_synthesis_is_identifier_free_and_read_only() -> None:
    if not OUTPUT.is_dir():
        pytest.skip("Evidence synthesis has not been generated in this checkout")
    prohibited = {"participant_hash", "match_hash", "puuid", "riot_id", "api_key"}
    for path in OUTPUT.glob("*.csv"):
        assert not prohibited.intersection(map(str.lower, pd.read_csv(path, nrows=0).columns))
    manifest = json.loads((OUTPUT / "evidence_synthesis_manifest.json").read_text())
    assert manifest["status"] == "PASS"
    assert manifest["production_artifact_hashes_unchanged"] is True
    assert manifest["production_model_retrained"] is False
    assert manifest["test_based_tuning"] is False
    assert manifest["rank_in_primary_matrix"] is False
    assert manifest["augment_analysis"] == "not_generated_schema_absent"

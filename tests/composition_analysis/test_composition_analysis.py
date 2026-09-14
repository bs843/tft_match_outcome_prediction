from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.composition_analysis.statistics import (association_table, odds_ratio_interval,
                                                        readable_name, validate_association_output,
                                                        wilson_interval)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "post_model" / "composition_analysis_v1"


def _participants() -> pd.DataFrame:
    return pd.DataFrame({"row_id": range(8), "top4": [1, 1, 1, 1, 0, 0, 0, 0],
                         "placement": range(1, 9)})


def test_global_baseline_and_counts_reconcile() -> None:
    frame = _participants()
    assert len(frame) == 8
    assert frame["top4"].mean() == .5
    assert set(frame["placement"]) == set(range(1, 9))


def test_uplift_and_minimum_support_are_correct() -> None:
    participants = _participants()
    exposures = participants.iloc[[0, 1, 4]].assign(feature="A")
    result = association_table(exposures, participants, feature_columns=["feature"],
                               global_rate=.5, minimum_support=4).iloc[0]
    assert np.isclose(result["top_four_rate"], 2 / 3)
    assert np.isclose(result["top_four_uplift_pp"], 100 * (2 / 3 - .5))
    assert not bool(result["meets_minimum_support"])


def test_odds_ratio_calculation_matches_two_by_two_table() -> None:
    odds, lower, upper = odds_ratio_interval(30, 10, 20, 40)
    assert np.isclose(odds, 6.0)
    assert 0 < lower < odds < upper


def test_wilson_interval_and_rates_are_bounded() -> None:
    lower, upper = wilson_interval(8, 10)
    assert 0 <= lower <= .8 <= upper <= 1
    participants = _participants()
    exposures = participants.assign(feature="all")
    result = association_table(exposures, participants, feature_columns=["feature"],
                               global_rate=.5, minimum_support=1)
    validate_association_output(result)


def test_readable_names_remove_riot_prefixes() -> None:
    assert readable_name("TFT17_AurelionSol") == "Aurelion Sol"
    assert readable_name("TFT_Item_GargoyleStoneplate") == "Gargoyle Stoneplate"


def test_generated_exports_contain_no_prohibited_identifiers() -> None:
    if not OUTPUT.is_dir():
        return
    prohibited = {"participant_hash", "match_hash", "puuid", "riot_id", "api_key"}
    for path in OUTPUT.rglob("*.csv"):
        assert not prohibited.intersection(map(str.lower, pd.read_csv(path, nrows=0).columns))


def test_generated_cohort_and_target_contract() -> None:
    if not OUTPUT.is_dir():
        return
    baseline = pd.read_csv(OUTPUT / "tables" / "global_baseline.csv").iloc[0]
    assert baseline["participant_observations"] == 40_000
    assert baseline["unique_matches"] == 5_000
    assert baseline["top_four_rate"] == .5
    assert baseline["target_definition"] == "top4 = placement <= 4"


def test_generated_manifest_confirms_frozen_bytes_unchanged() -> None:
    if not OUTPUT.is_dir():
        return
    manifest = json.loads((OUTPUT / "composition_analysis_manifest.json").read_text())
    assert manifest["status"] == "PASS"
    assert manifest["production_artifact_hashes_before_and_after_equal"] is True
    assert manifest["model_retraining"] is False
    assert manifest["test_set_tuning"] is False

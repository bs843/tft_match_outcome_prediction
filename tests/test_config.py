from __future__ import annotations

from copy import deepcopy

import pytest

from src.config import ConfigurationError, _validate_config, load_config


def test_repository_config_is_strictly_valid() -> None:
    config = load_config()
    assert config["scope"]["allowed_patches"] == ["17.8"]
    assert config["scope"]["allowed_client_patches"] == ["16.15"]
    assert config["validation"]["reject_on_wrong_patch"] is True
    assert config["project"]["dataset_version"] == "tft_set17_patch17_8_v8"
    assert config["project"]["schema_version"] == "1.2.1"
    assert config["provenance"]["configuration_version"] == "tft-patch17.8-final-collection-v7"
    assert config["provenance"]["required_git_tag"] == "tft-patch17.8-final-collection-v7"
    assert config["api"]["authentication_recovery_enabled"] is True
    assert config["provenance"]["enforce_git_integrity_for_main"] is True
    assert config["sampling"]["target_accepted_matches"] == 5000
    assert config["sampling"]["seed_players_per_stratum"] == 160
    assert config["sampling"]["match_history_pages_per_seed"] == 3
    assert config["sampling"]["candidate_pool_capacity_target"] == 25000
    expected = {
        "GOLD": ["I", "II", "III", "IV"],
        "PLATINUM": ["I", "II", "III", "IV"],
        "EMERALD": ["I", "II", "III", "IV"],
        "DIAMOND": ["I", "II", "III", "IV"],
        "MASTER": [],
    }
    assert {row["tier"]: row["divisions"] for row in config["sampling"]["rank_strata"]} == expected
    assert all(row["pages_per_division"] == (1 if row["tier"] == "MASTER" else 3)
               for row in config["sampling"]["rank_strata"])
    assert config["paths"]["rank_snapshots"].endswith("rank_snapshots.csv")
    required_rank_exclusions = {
        "rank_snapshot_tier", "rank_snapshot_division", "rank_snapshot_league_points",
        "rank_snapshot_wins", "rank_snapshot_losses", "rank_snapshot_queue_type",
        "rank_snapshot_retrieved_at", "rank_snapshot_status",
    }
    assert required_rank_exclusions.issubset(config["features"]["prohibited_columns"])
    assert config["scope"]["allowed_end_results"] == ["GameComplete"]
    assert config["validation"]["reject_on_wrong_end_result"] is True
    assert [(row["platform_region"], row["regional_route"]) for row in config["platforms"]
            if row["enabled"]] == [
        ("ME1", "EUROPE"), ("EUW1", "EUROPE"), ("EUN1", "EUROPE"),
        ("NA1", "AMERICAS"), ("KR", "ASIA"),
    ]
    assert all(row["rank_strata_profile"] == "GOLD_TO_MASTER" and
               row["ladder_pages_per_division"] == 3 and row["apex_pages"] == 1 and
               row["seed_players_per_page"] == 160 and
               row["match_history_pages_per_seed"] == 3
               for row in config["platforms"] if row["enabled"])


def test_misspelled_or_empty_patch_contract_fails() -> None:
    config = load_config()
    broken = deepcopy(config)
    broken["scope"]["allowed_patches"] = []
    with pytest.raises(ConfigurationError, match="allowed_patches"):
        _validate_config(broken)


def test_split_fractions_must_sum_to_one() -> None:
    config = load_config()
    broken = deepcopy(config)
    broken["splits"]["test_fraction"] = 0.2
    with pytest.raises(ConfigurationError, match="split fractions"):
        _validate_config(broken)

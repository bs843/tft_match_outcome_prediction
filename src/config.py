from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigurationError(RuntimeError):
    """Raised when the frozen study contract is incomplete or inconsistent."""


REQUIRED_SECTIONS = {
    "project",
    "provenance",
    "scope",
    "platforms",
    "sampling",
    "api",
    "privacy",
    "validation",
    "features",
    "splits",
    "model",
    "paths",
}

REQUIRED_SCOPE_KEYS = {
    "study_patch",
    "allowed_patches",
    "allowed_client_patches",
    "set_number",
    "allowed_queue_ids",
    "expected_participants",
}

SUPPORTED_PLATFORM_ROUTES = {
    "BR1": "AMERICAS", "EUN1": "EUROPE", "EUW1": "EUROPE", "JP1": "ASIA",
    "KR": "ASIA", "LA1": "AMERICAS", "LA2": "AMERICAS", "ME1": "EUROPE",
    "NA1": "AMERICAS", "OC1": "SEA", "PH2": "SEA", "RU": "EUROPE",
    "SG2": "SEA", "TH2": "SEA", "TR1": "EUROPE", "TW2": "SEA", "VN2": "SEA",
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load environment and YAML configuration, then enforce the study contract."""
    load_dotenv(PROJECT_ROOT / ".env")
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration root must be a mapping")

    missing = REQUIRED_SECTIONS.difference(config)
    if missing:
        raise ConfigurationError(
            f"Missing configuration section(s): {', '.join(sorted(missing))}"
        )
    _validate_config(config)
    config["_config_path"] = str(path.resolve())
    return config


def _validate_config(config: dict[str, Any]) -> None:
    project = _mapping(config, "project")
    if project.get("dataset_version") != "tft_set17_patch17_8_v7":
        raise ConfigurationError("Routing-amended dataset_version must be tft_set17_patch17_8_v7")
    if str(project.get("schema_version", "")) != "1.2.1":
        raise ConfigurationError("Routing-amended schema_version must be 1.2.1")

    provenance = _mapping(config, "provenance")
    if provenance.get("configuration_version") != "tft-patch17.8-final-collection-v6":
        raise ConfigurationError(
            "provenance.configuration_version must identify final collection v6"
        )
    if provenance.get("required_git_tag") != "tft-patch17.8-final-collection-v6":
        raise ConfigurationError(
            "provenance.required_git_tag must be tft-patch17.8-final-collection-v6"
        )
    if provenance.get("enforce_git_integrity_for_main") is not True:
        raise ConfigurationError("provenance.enforce_git_integrity_for_main must be true")

    scope = _mapping(config, "scope")
    missing_scope = REQUIRED_SCOPE_KEYS.difference(scope)
    if missing_scope:
        raise ConfigurationError(
            f"Missing scope key(s): {', '.join(sorted(missing_scope))}"
        )

    study_patch = str(scope["study_patch"]).strip()
    allowed_patches = [str(value).strip() for value in _list(scope, "allowed_patches")]
    if not study_patch or allowed_patches != [study_patch]:
        raise ConfigurationError(
            "scope.allowed_patches must contain exactly scope.study_patch"
        )
    if study_patch != "17.8":
        raise ConfigurationError("This frozen study configuration must target patch 17.8")
    client_patches = [str(value).strip() for value in _list(scope, "allowed_client_patches")]
    if client_patches != ["16.15"]:
        raise ConfigurationError(
            "scope.allowed_client_patches must contain the observed API client patch 16.15"
        )
    if int(scope["set_number"]) != 17:
        raise ConfigurationError("This study requires TFT set_number 17")
    if not _list(scope, "allowed_queue_ids"):
        raise ConfigurationError("scope.allowed_queue_ids cannot be empty")
    if int(scope["expected_participants"]) != 8:
        raise ConfigurationError("Standard ranked TFT requires eight participants")
    platforms = enabled_platforms(config)
    if len(platforms) < 2:
        raise ConfigurationError("Multi-region collection requires at least two enabled platforms")
    for platform in platforms:
        platform_region = platform["platform_region"]
        if SUPPORTED_PLATFORM_ROUTES.get(platform_region) != platform["regional_route"]:
            raise ConfigurationError(
                f"Unsupported platform/regional route mapping: {platform_region} -> "
                f"{platform['regional_route']}"
            )
        if platform.get("rank_strata_profile") != "GOLD_TO_MASTER":
            raise ConfigurationError("Every platform must use GOLD_TO_MASTER")
        if int(platform.get("ladder_pages_per_division", 0)) != 3:
            raise ConfigurationError("Every platform requires three non-apex ladder pages")
        if int(platform.get("apex_pages", 0)) != 1:
            raise ConfigurationError("Every platform requires one MASTER page")
        if int(platform.get("seed_players_per_page", 0)) != 160:
            raise ConfigurationError("Every platform requires 160 seeds per populated page")
        if int(platform.get("match_history_pages_per_seed", 0)) != 3:
            raise ConfigurationError("Every platform requires three history pages per seed")

    validation = _mapping(config, "validation")
    for key in (
        "reject_on_wrong_patch",
        "reject_on_wrong_set",
        "reject_on_wrong_queue",
    ):
        if validation.get(key) is not True:
            raise ConfigurationError(f"validation.{key} must be true")
    if [str(value) for value in scope.get("allowed_end_results", [])] != ["GameComplete"]:
        raise ConfigurationError("scope.allowed_end_results must contain exactly GameComplete")
    if validation.get("reject_on_wrong_end_result") is not True:
        raise ConfigurationError("validation.reject_on_wrong_end_result must be true")
    if str(validation.get("validation_code_version", "")) != "1.1.0":
        raise ConfigurationError("Multi-region validation_code_version must be 1.1.0")

    prohibited = set(_list(_mapping(config, "features"), "prohibited_columns"))
    required_rank_exclusions = {
        "rank_snapshot_tier", "rank_snapshot_division", "rank_snapshot_league_points",
        "rank_snapshot_wins", "rank_snapshot_losses", "rank_snapshot_queue_type",
        "rank_snapshot_retrieved_at", "rank_snapshot_status",
        "rank_snapshot_error_class",
    }
    missing_rank_exclusions = required_rank_exclusions.difference(prohibited)
    if missing_rank_exclusions:
        raise ConfigurationError(
            "features.prohibited_columns is missing rank snapshot fields: "
            + ", ".join(sorted(missing_rank_exclusions))
        )
    missing_region_exclusions = {"platform_region", "regional_route", "region"}.difference(prohibited)
    if missing_region_exclusions:
        raise ConfigurationError(
            "features.prohibited_columns must exclude region provenance: "
            + ", ".join(sorted(missing_region_exclusions))
        )

    sampling = _mapping(config, "sampling")
    if int(sampling.get("match_history_page_size", 0)) not in range(1, 101):
        raise ConfigurationError("sampling.match_history_page_size must be 1..100")
    if int(sampling.get("match_history_pages_per_seed", 0)) < 1:
        raise ConfigurationError("sampling.match_history_pages_per_seed must be positive")
    if int(sampling["match_history_pages_per_seed"]) != 3:
        raise ConfigurationError("Expanded discovery requires three match-history pages per seed")
    for key in ("pilot_accepted_matches", "target_accepted_matches"):
        if int(sampling.get(key, 0)) < 1:
            raise ConfigurationError(f"sampling.{key} must be positive")
    if int(sampling["target_accepted_matches"]) != 5000:
        raise ConfigurationError("The frozen main collection target must be exactly 5000 matches")
    if int(sampling.get("seed_players_per_stratum", 0)) != 160:
        raise ConfigurationError("Expanded breadth requires 160 seed players per populated page")
    if int(sampling.get("candidate_pool_capacity_target", 0)) != 25000:
        raise ConfigurationError("Candidate discovery capacity gate must be 25000")
    strata = sampling.get("rank_strata", [])
    expected_strata = {
        "GOLD": ["I", "II", "III", "IV"],
        "PLATINUM": ["I", "II", "III", "IV"],
        "EMERALD": ["I", "II", "III", "IV"],
        "DIAMOND": ["I", "II", "III", "IV"],
        "MASTER": [],
    }
    observed_strata = {
        str(row.get("tier", "")).upper(): [str(value).upper() for value in row.get("divisions", [])]
        for row in strata if isinstance(row, dict)
    }
    if observed_strata != expected_strata:
        raise ConfigurationError("The frozen ladder rank strata must remain unchanged")
    for row in strata:
        expected_pages = 1 if str(row.get("tier", "")).upper() == "MASTER" else 3
        if int(row.get("pages_per_division", 0)) != expected_pages:
            raise ConfigurationError("Non-apex ladder strata require three pages; MASTER requires one")

    splits = _mapping(config, "splits")
    fractions = [float(splits.get(key, 0)) for key in (
        "train_fraction", "validation_fraction", "test_fraction"
    )]
    if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ConfigurationError("split fractions must be positive and sum to 1")

    paths = _mapping(config, "paths")
    required_paths = {
        "input_seed_players", "seed_players", "match_id_sources", "unique_match_ids",
        "collection_status", "player_sampling_report", "player_sampling_pages",
        "candidate_discovery_report",
        "accepted_match_ids", "rejected_match_ids",
        "rank_snapshots",
        "raw_matches", "manifests", "interim", "processed", "splits",
        "models", "reports", "request_log", "quality_report",
        "validated", "validation_reports", "collection_reports",
    }
    missing_paths = required_paths.difference(paths)
    if missing_paths:
        raise ConfigurationError(
            f"Missing paths key(s): {', '.join(sorted(missing_paths))}"
        )
    if paths["rank_snapshots"] != "data/private/metadata/rank_snapshots.csv":
        raise ConfigurationError(
            "paths.rank_snapshots must be data/private/metadata/rank_snapshots.csv"
        )


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"{key} must be a mapping")
    return value


def _list(mapping: dict[str, Any], key: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ConfigurationError(f"{key} must be a list")
    return value


def enabled_platforms(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized enabled platform mappings in configured order."""
    value = config.get("platforms")
    if not isinstance(value, list):
        raise ConfigurationError("platforms must be a list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in value:
        if not isinstance(row, dict):
            raise ConfigurationError("Each platforms entry must be a mapping")
        if not bool(row.get("enabled")):
            continue
        normalized = dict(row)
        normalized["platform_region"] = str(row.get("platform_region", "")).upper().strip()
        normalized["regional_route"] = str(row.get("regional_route", "")).upper().strip()
        if not normalized["platform_region"] or not normalized["regional_route"]:
            raise ConfigurationError("Enabled platforms require platform_region and regional_route")
        if normalized["platform_region"] in seen:
            raise ConfigurationError(f"Duplicate enabled platform {normalized['platform_region']}")
        seen.add(normalized["platform_region"])
        result.append(normalized)
    return result


def route_for_platform(config: dict[str, Any], platform_region: str) -> str:
    platform = str(platform_region).upper().strip()
    for row in enabled_platforms(config):
        if row["platform_region"] == platform:
            return row["regional_route"]
    raise ConfigurationError(f"Platform {platform or '<blank>'} is not enabled")


def project_path(config: dict[str, Any], path_key: str) -> Path:
    value = config.get("paths", {}).get(path_key)
    if not value:
        raise ConfigurationError(f"Missing paths.{path_key} in configuration")
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def required_environment_value(config: dict[str, Any], dotted_key: str) -> str:
    section_name, key_name = dotted_key.split(".", 1)
    variable_name = config.get(section_name, {}).get(key_name)
    if not variable_name:
        raise ConfigurationError(f"Configuration key {dotted_key} is not set")
    value = os.getenv(str(variable_name))
    if not value:
        raise ConfigurationError(
            f"Environment variable {variable_name} is missing. Copy .env.example to .env."
        )
    return value


def ensure_project_directories(config: dict[str, Any]) -> None:
    for key in (
        "raw_matches", "manifests", "interim", "processed", "splits", "models",
        "reports", "validated", "validation_reports", "collection_reports",
    ):
        project_path(config, key).mkdir(parents=True, exist_ok=True)
    for key in (
        "input_seed_players", "seed_players", "match_id_sources", "unique_match_ids",
        "collection_status", "player_sampling_report", "player_sampling_pages",
        "candidate_discovery_report",
        "accepted_match_ids", "rejected_match_ids",
        "rank_snapshots",
        "request_log", "quality_report",
    ):
        project_path(config, key).parent.mkdir(parents=True, exist_ok=True)


def normalise_route(value: str) -> str:
    if not value or not value.strip():
        raise ConfigurationError("A Riot routing value cannot be empty")
    return value.strip().lower()

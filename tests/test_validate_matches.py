from __future__ import annotations

from copy import deepcopy

from src.utils import parse_patch, placement_group
from src.validate_matches import validate_match_payload


def sample_config() -> dict:
    return {
        "scope": {
            "study_patch": "17.8", "allowed_patches": ["17.8"],
            "allowed_client_patches": ["16.15"], "set_number": 17,
            "allowed_queue_ids": [1100], "allowed_game_types": ["standard"],
            "allowed_end_results": ["GameComplete"], "expected_participants": 8,
            "start_time": 1_700_000_000, "end_time": 1_800_000_000,
        },
        "validation": {
            "reject_on_wrong_set": True, "reject_on_wrong_patch": True,
            "reject_on_wrong_queue": True, "reject_on_wrong_game_type": True,
            "reject_on_wrong_end_result": True, "reject_outside_collection_window": True,
            "reject_on_unexpected_participant_count": True,
            "reject_on_invalid_participant": True, "require_complete_placement_set": True,
            "require_nonempty_units": True, "require_traits_list": True,
            "require_metadata_participant_match": True,
        },
    }


def sample_match() -> dict:
    participants = [{
        "puuid": f"player-{placement}", "placement": placement, "level": 8,
        "units": [{"character_id": "TFT17_TestUnit", "tier": 2, "rarity": 3,
                   "items": [], "itemNames": []}],
        "traits": [{"name": "TFT17_TestTrait", "num_units": 2, "style": 1,
                    "tier_current": 1, "tier_total": 3}],
    } for placement in range(1, 9)]
    puuids = [participant["puuid"] for participant in participants]
    return {
        "metadata": {"data_version": "6", "match_id": "ME1_TEST_MATCH", "participants": puuids},
        "info": {
            "game_version": "Linux Version 16.15.801.3452 (Jul 30 2026/15:55:26) [PUBLIC] <Releases/16.15>", "queue_id": 1100,
            "tft_set_number": 17, "tft_game_type": "standard",
            "game_datetime": 1_750_000_000_000,
            "endOfGameResult": "GameComplete",
            "participants": participants,
        },
    }


def test_valid_patch_17_8_match_is_accepted() -> None:
    result = validate_match_payload(sample_match(), sample_config(), expected_match_id="ME1_TEST_MATCH")
    assert result.accepted is True
    assert result.study_patch == "17.8"
    assert result.client_patch == "16.15"
    assert result.participant_count == 8
    assert result.end_result == "GameComplete"
    assert result.end_result_source == "info.endOfGameResult"


def test_missing_real_end_result_is_rejected() -> None:
    payload = deepcopy(sample_match())
    del payload["info"]["endOfGameResult"]
    result = validate_match_payload(payload, sample_config())
    assert result.accepted is False
    assert "wrong_or_missing_end_result" in result.rejection_reasons


def test_wrong_patch_is_rejected() -> None:
    payload = deepcopy(sample_match())
    payload["info"]["game_version"] = "Linux Version 16.14.999.1 <Releases/16.14>"
    result = validate_match_payload(payload, sample_config())
    assert result.accepted is False
    assert "wrong_or_missing_client_patch" in result.rejection_reasons


def test_metadata_participants_must_match_info() -> None:
    payload = deepcopy(sample_match())
    payload["metadata"]["participants"][0] = "different-player"
    result = validate_match_payload(payload, sample_config())
    assert "metadata_participant_mismatch" in result.rejection_reasons


def test_wrong_queue_and_missing_units_are_rejected() -> None:
    payload = deepcopy(sample_match())
    payload["info"]["queue_id"] = 1090
    payload["info"]["participants"][0]["units"] = []
    result = validate_match_payload(payload, sample_config())
    assert {"wrong_or_missing_queue", "invalid_participant_record"}.issubset(result.rejection_reasons)


def test_wrong_set_is_rejected() -> None:
    payload = deepcopy(sample_match())
    payload["info"]["tft_set_number"] = 16
    result = validate_match_payload(payload, sample_config())
    assert result.accepted is False
    assert "wrong_or_missing_set" in result.rejection_reasons


def test_incomplete_match_is_rejected() -> None:
    payload = deepcopy(sample_match())
    payload["info"]["participants"].pop()
    payload["metadata"]["participants"].pop()
    result = validate_match_payload(payload, sample_config())
    assert result.accepted is False
    assert {"unexpected_participant_count", "incomplete_or_duplicate_placements"}.issubset(
        result.rejection_reasons
    )


def test_patch_parser_and_placement_group() -> None:
    assert parse_patch("Linux Version 16.15.801.3452 <Releases/16.15>") == "16.15"
    assert parse_patch("Linux Version malformed") == ""
    assert placement_group(1) == "1-2"
    assert placement_group(4) == "3-4"
    assert placement_group(8) == "7-8"

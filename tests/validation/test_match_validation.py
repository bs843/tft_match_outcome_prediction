from __future__ import annotations

from src.validation.validate_matches import validate_match

from .helpers import clone, rules, valid_match

HASH = lambda value: f"hash-{value}"


def check(payload, *, seen=None):
    return validate_match(payload, expected_file_match_id=payload.get("metadata", {}).get("match_id", "ME1_TEST"),
                          hash_identifier=HASH, rules=rules(),
                          seen_match_ids=seen if seen is not None else set())


def codes(result):
    return {issue.reason_code for issue in result["issues"]}


def test_valid_eight_player_match() -> None:
    result = check(valid_match())
    assert result["valid"] is True
    assert result["patch"]["classification"] == "patch_17_8"
    assert len(result["participants"]) == 8


def test_duplicate_match_id_is_critical() -> None:
    seen: set[str] = set()
    first = check(valid_match(), seen=seen)
    second = check(valid_match(), seen=seen)
    assert first["valid"] is True
    assert "duplicate_match_id" in codes(second)


def test_invalid_and_missing_placements_are_detected() -> None:
    invalid = valid_match()
    invalid["info"]["participants"][0]["placement"] = 9
    assert {"invalid_placement", "invalid_placement_structure"}.issubset(codes(check(invalid)))
    missing = valid_match()
    del missing["info"]["participants"][0]["placement"]
    assert {"missing_required_field", "invalid_placement"}.issubset(codes(check(missing)))


def test_non_target_and_malformed_patch_are_detected() -> None:
    wrong = valid_match()
    wrong["info"]["game_version"] = "Linux Version 16.14.1"
    assert "non_17_8_patch" in codes(check(wrong))
    malformed = valid_match()
    malformed["info"]["game_version"] = "not-a-version"
    assert "unknown_patch" in codes(check(malformed))


def test_blank_end_result_is_critical() -> None:
    payload = valid_match()
    payload["info"]["endOfGameResult"] = ""
    result = check(payload)
    assert "invalid_end_result" in codes(result)
    assert result["end_result_blank"] is True


def test_missing_required_and_malformed_nested_structures() -> None:
    payload = valid_match()
    participant = payload["info"]["participants"][0]
    del participant["level"]
    participant["units"] = [{"items": "not-a-list", "itemNames": []}]
    participant["traits"] = ["not-an-object"]
    detected = codes(check(payload))
    assert {"missing_required_field", "invalid_level", "malformed_items", "malformed_trait"}.issubset(detected)

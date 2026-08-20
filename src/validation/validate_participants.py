from __future__ import annotations

from collections import Counter
from typing import Any

from ..utils import safe_int
from .validation_report import Issue

PARTICIPANT_FIELDS = [
    "puuid", "placement", "level", "units", "traits", "gold_left", "last_round",
    "players_eliminated", "total_damage_to_players", "time_eliminated", "augments",
]
REQUIRED_PARTICIPANT_FIELDS = {"puuid", "placement", "level", "units", "traits"}


def validate_participant(
    participant: Any, *, match_hash: str, participant_hash: str,
    placement_min: int, placement_max: int, level_min: int, level_max: int,
) -> tuple[list[Issue], Counter[str], dict[str, Any]]:
    issues: list[Issue] = []
    missing: Counter[str] = Counter()
    derived: dict[str, Any] = {"placement": None, "top_four": None, "valid": True}
    if not isinstance(participant, dict):
        issues.append(Issue("critical", "participant", "participant_not_object",
                            "Participant record is not an object", match_hash, participant_hash))
        derived["valid"] = False
        return issues, missing, derived

    for field in PARTICIPANT_FIELDS:
        value = participant.get(field)
        if value is None or value == "":
            missing[field] += 1
            if field in REQUIRED_PARTICIPANT_FIELDS:
                issues.append(Issue("critical", "participant", "missing_required_field",
                                    f"Required participant field {field} is missing",
                                    match_hash, participant_hash, field))

    placement = safe_int(participant.get("placement"))
    derived["placement"] = placement
    derived["top_four"] = placement is not None and placement <= 4
    if placement is None or not placement_min <= placement <= placement_max:
        issues.append(Issue("critical", "participant", "invalid_placement",
                            "Placement is missing or outside the configured range",
                            match_hash, participant_hash, "placement"))

    level = safe_int(participant.get("level"))
    if level is None or not level_min <= level <= level_max:
        issues.append(Issue("critical", "participant", "invalid_level",
                            "Player level is missing or outside the sensible range",
                            match_hash, participant_hash, "level"))

    units = participant.get("units")
    if not isinstance(units, list):
        issues.append(Issue("critical", "participant", "malformed_units",
                            "Final units must be a list", match_hash, participant_hash, "units"))
    else:
        for unit in units:
            if not isinstance(unit, dict):
                issues.append(Issue("critical", "participant", "malformed_unit",
                                    "A unit entry is not an object", match_hash, participant_hash, "units"))
                break
            if not isinstance(unit.get("items", []), list) or not isinstance(unit.get("itemNames", []), list):
                issues.append(Issue("critical", "participant", "malformed_items",
                                    "Unit item representations must be lists",
                                    match_hash, participant_hash, "units.items"))
                break

    traits = participant.get("traits")
    if not isinstance(traits, list):
        issues.append(Issue("critical", "participant", "malformed_traits",
                            "Traits must be a list", match_hash, participant_hash, "traits"))
    elif any(not isinstance(trait, dict) for trait in traits):
        issues.append(Issue("critical", "participant", "malformed_trait",
                            "A trait entry is not an object", match_hash, participant_hash, "traits"))

    augments = participant.get("augments")
    if augments is not None and not isinstance(augments, list):
        issues.append(Issue("warning", "participant", "malformed_optional_augments",
                            "Optional augments field is not a list",
                            match_hash, participant_hash, "augments"))
    derived["valid"] = not any(issue.severity == "critical" for issue in issues)
    return issues, missing, derived

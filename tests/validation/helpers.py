from __future__ import annotations

from copy import deepcopy


def rules() -> dict:
    return {
        "allowed_client_patches": ["16.15"], "allowed_queue_ids": [1100],
        "allowed_game_types": ["standard"], "allowed_end_results": ["GameComplete"],
        "expected_participants_per_match": 8, "valid_placement_min": 1,
        "valid_placement_max": 8, "sensible_level_min": 1, "sensible_level_max": 15,
        "sensible_game_duration_max_seconds": 14400, "fail_on_patch_mismatch": True,
        "fail_on_end_result_mismatch": True, "rank_snapshot_required": False,
        "expected_patch": "17.8",
    }


def valid_match(match_id: str = "ME1_TEST") -> dict:
    participants = [{
        "puuid": f"private-{placement}", "placement": placement, "level": 8,
        "gold_left": 1, "last_round": 30, "players_eliminated": 1,
        "total_damage_to_players": 50, "time_eliminated": 1000,
        "units": [{"character_id": "TFT17_UNIT", "tier": 2,
                   "items": [1], "itemNames": ["TFT17_ITEM"]}],
        "traits": [{"name": "TFT17_TRAIT", "tier_current": 1}],
        "augments": [],
    } for placement in range(1, 9)]
    return {
        "metadata": {"match_id": match_id, "data_version": "6",
                     "participants": [row["puuid"] for row in participants]},
        "info": {
            "game_version": "Linux Version 16.15.1 <Releases/16.15>",
            "queue_id": 1100, "tft_set_number": 17, "tft_game_type": "standard",
            "game_datetime": 1_786_000_000_000, "game_length": 1800,
            "endOfGameResult": "GameComplete", "participants": participants,
        },
    }


def clone(value):
    return deepcopy(value)

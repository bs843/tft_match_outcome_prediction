from __future__ import annotations

import json

import pandas as pd

from .dataset import validate_participant_dataset


def synthetic_participant_dataset(matches: int = 20) -> pd.DataFrame:
    rows = []
    for match_index in range(matches):
        for placement in range(1, 9):
            top = placement <= 4
            units = [
                {"name": f"TFT17_Unit_{(placement + offset + match_index) % 12:02d}",
                 "stars": 2 if top else 1,
                 "items": [f"TFT_Item_{(placement + offset) % 8:02d}"] if offset < 3 else []}
                for offset in range(7 if top else 6)
            ]
            traits = [{"name": f"TFT17_Trait_{placement % 5}", "num_units": 4,
                       "style": 2 if top else 1, "tier_current": 2 if top else 1}]
            items = [item for unit in units for item in unit["items"]]
            rows.append({
                "collection_run_id": "SYNTHETIC",
                "match_hash": f"synthetic_match_{match_index:03d}",
                "participant_hash": f"synthetic_participant_{match_index:03d}_{placement}",
                "participant_slot": placement - 1,
                "game_datetime": "SYNTHETIC",
                "study_patch": "17.8",
                "set_number": 17,
                "queue_id": 1100,
                "platform_region": "SYNTHETIC",
                "regional_route": "SYNTHETIC",
                "placement": placement,
                "top4": int(top),
                "level": 9 if top else 7,
                "composition_size": len(units),
                "units_json": json.dumps(units, sort_keys=True, separators=(",", ":")),
                "traits_json": json.dumps(traits, sort_keys=True, separators=(",", ":")),
                "items_json": json.dumps(items, sort_keys=True, separators=(",", ":")),
                "rank_tier": "SYNTHETIC",
                "rank_division": "SYNTHETIC",
                "rank_status": "synthetic",
            })
    frame = pd.DataFrame(rows)
    validate_participant_dataset(frame)
    return frame

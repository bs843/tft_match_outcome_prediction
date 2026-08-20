from __future__ import annotations

from typing import Any

from ..utils import parse_patch


def classify_patch(payload: dict[str, Any], allowed_client_patches: set[str]) -> dict[str, str]:
    info = payload.get("info") if isinstance(payload, dict) else None
    game_version = str(info.get("game_version", "")) if isinstance(info, dict) else ""
    client_patch = parse_patch(game_version)
    if not client_patch:
        classification = "unknown_patch"
    elif client_patch in allowed_client_patches:
        classification = "patch_17_8"
    else:
        classification = "non_17_8_patch"
    return {
        "classification": classification,
        "client_patch": client_patch,
        "game_version": game_version,
    }

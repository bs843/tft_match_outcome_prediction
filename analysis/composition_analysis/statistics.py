from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def readable_name(value: str) -> str:
    """Convert Riot data identifiers to readable, deterministic labels."""
    text = str(value)
    for prefix in ("TFT17_", "TFT_Item_", "TFT_Augment_", "TFT_"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
            break
    text = re.sub(r"Trait$", "", text)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = text.replace("_", " ").strip()
    return re.sub(r"\s+", " ", text).title()


def wilson_interval(successes: int, total: int, confidence: float = .95) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def odds_ratio_interval(exposed_top: int, exposed_bottom: int, unexposed_top: int,
                        unexposed_bottom: int, confidence: float = .95) -> tuple[float, float, float]:
    cells = np.asarray([exposed_top, exposed_bottom, unexposed_top, unexposed_bottom], dtype=float)
    if np.any(cells == 0):
        cells += .5
    a, b, c, d = cells
    log_or = np.log((a * d) / (b * c))
    se = np.sqrt(np.sum(1 / cells))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    return float(np.exp(log_or)), float(np.exp(log_or - z * se)), float(np.exp(log_or + z * se))


def benjamini_hochberg(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    result = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return result
    positions = np.where(valid)[0]
    order = np.argsort(p[valid])
    ranked = p[valid][order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result[positions[order]] = adjusted
    return result


def association_table(exposures: pd.DataFrame, participants: pd.DataFrame, *,
                      feature_columns: list[str], global_rate: float,
                      minimum_support: int) -> pd.DataFrame:
    """Summarise participant-level binary feature exposures without identifiers."""
    required = {"row_id", "top4", "placement", *feature_columns}
    if not required.issubset(exposures.columns):
        raise ValueError(f"Exposure table missing columns: {sorted(required - set(exposures.columns))}")
    total_top = int(participants["top4"].sum())
    total_bottom = len(participants) - total_top
    rows = []
    for keys, group in exposures.drop_duplicates(["row_id", *feature_columns]).groupby(
        feature_columns, dropna=False, sort=True
    ):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = len(group); top = int(group["top4"].sum()); bottom = n - top
        lower, upper = wilson_interval(top, n)
        unexposed_top = total_top - top; unexposed_bottom = total_bottom - bottom
        odds, odds_low, odds_high = odds_ratio_interval(top, bottom, unexposed_top, unexposed_bottom)
        _, fisher_p = stats.fisher_exact([[top, bottom], [unexposed_top, unexposed_bottom]])
        row = dict(zip(feature_columns, keys, strict=True))
        row.update({
            "appearances": n, "pick_rate": n / len(participants),
            "top_four_count": top, "top_four_rate": top / n,
            "bottom_four_count": bottom, "average_placement": group["placement"].mean(),
            "median_placement": group["placement"].median(),
            "top_four_uplift_pp": 100 * (top / n - global_rate),
            "top_four_rate_ci95_low": lower, "top_four_rate_ci95_high": upper,
            "odds_ratio": odds, "odds_ratio_ci95_low": odds_low,
            "odds_ratio_ci95_high": odds_high, "fisher_p_value": fisher_p,
            "minimum_support": minimum_support, "meets_minimum_support": n >= minimum_support,
            "rare_feature_warning": n < minimum_support,
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr_bh_q_value"] = benjamini_hochberg(result["fisher_p_value"])
    return result.sort_values(["meets_minimum_support", "appearances"], ascending=[False, False])


def validate_association_output(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    for column in ("top_four_rate", "pick_rate", "top_four_rate_ci95_low", "top_four_rate_ci95_high"):
        if not frame[column].dropna().between(0, 1).all():
            raise RuntimeError(f"Invalid probability in {column}")
    if (frame["appearances"] <= 0).any():
        raise RuntimeError("Association output contains non-positive support")

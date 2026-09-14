from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy import stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from analysis.post_model.common import sha256_file, utc_now, write_table, write_text
from analysis.src.common import safe_feature_name
from analysis.thesis_extension.run import feature_group
from src.config import PROJECT_ROOT

from .statistics import (association_table, benjamini_hochberg, readable_name,
                         validate_association_output, wilson_interval)


VERSION = "1.0.0"
RUN_ID = "analysis-20260825T190121Z-1b3eaaff"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "post_model" / "composition_analysis_v1"
LEGACY_REPOSITORY = Path(
    r"C:\Users\bakan\Documents\Codex\2026-08-06\review-this-repository-without-changing-anything\outputs\tft-thesis-data-collection"
)
SEED = 4275
UNIT_MINIMUM = 200
ITEM_MINIMUM = 200
TRAIT_MINIMUM = 200
STAR_MINIMUM = 100
INTERACTION_MINIMUM = 150
RANK_MINIMUM = 100
RANK_ORDER = ["GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER",
              "GRANDMASTER", "CHALLENGER", "SILVER", "BRONZE"]
PROHIBITED = {"participant_hash", "match_hash", "puuid", "riot_id", "api_key"}
ITEM_COMPONENTS_OR_PLACEHOLDERS = {
    "TFT_Item_BFSword", "TFT_Item_ChainVest", "TFT_Item_EmptyBag",
    "TFT_Item_FryingPan", "TFT_Item_GiantsBelt", "TFT_Item_NeedlesslyLargeRod",
    "TFT_Item_NegatronCloak", "TFT_Item_RecurveBow", "TFT_Item_SparringGloves",
    "TFT_Item_Spatula", "TFT_Item_TearOfTheGoddess",
}


def locate_production(explicit: Path | None = None) -> Path:
    candidates = ([explicit] if explicit else []) + [PROJECT_ROOT, LEGACY_REPOSITORY]
    for repository in candidates:
        if repository is None:
            continue
        root = repository / "outputs" / RUN_ID
        if (root / "data" / "train_participants.csv.gz").is_file() and (
            root / "modelling" / "models" / "development" / "top4_f5_xgboost.joblib"
        ).is_file():
            return root.resolve()
    raise RuntimeError("Full frozen production inputs are unavailable in known read-only locations")


def production_hashes(root: Path) -> dict[str, str]:
    rows = list(csv.DictReader((root / "artifact_manifest.csv").open(encoding="utf-8-sig")))
    hashes = {}
    for row in rows:
        path = root / row["relative_path"]
        if not path.is_file():
            raise RuntimeError(f"Manifested frozen artifact is missing: {path}")
        observed = sha256_file(path)
        if observed != row["sha256"]:
            raise RuntimeError(f"Frozen artifact hash mismatch: {row['relative_path']}")
        hashes[row["relative_path"]] = observed
    return hashes


def load_participants(root: Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "validation", "test"):
        frame = pd.read_csv(root / "data" / f"{split}_participants.csv.gz")
        frame.insert(0, "split", split); frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.insert(0, "row_id", np.arange(len(result), dtype=int))
    if len(result) != 40_000 or result["match_hash"].nunique() != 5_000:
        raise RuntimeError("Frozen cohort is not exactly 40,000 participants from 5,000 matches")
    if set(result["placement"].unique()) != set(range(1, 9)):
        raise RuntimeError("Placement contract differs from 1-8")
    if not np.array_equal(result["top4"].astype(bool), result["placement"].le(4)):
        raise RuntimeError("Binary target differs from placement <= 4")
    if (result.groupby("match_hash")["split"].nunique() != 1).any():
        raise RuntimeError("A match crosses frozen partitions")
    return result


def explode_features(participants: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    unit_rows: list[dict[str, Any]] = []; item_rows: list[dict[str, Any]] = []
    holder_rows: list[dict[str, Any]] = []; trait_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    for source in participants.itertuples(index=False):
        units = json.loads(source.units_json); traits = json.loads(source.traits_json)
        items = json.loads(source.items_json)
        base = {"row_id": source.row_id, "top4": int(source.top4),
                "placement": int(source.placement), "rank_tier": source.rank_tier,
                "rank_status": source.rank_status, "split": source.split}
        for unit in units:
            name = readable_name(unit["name"]); stars = int(unit.get("stars", 0) or 0)
            unit_rows.append({**base, "unit": name, "star_level": stars})
            for item in set(map(str, unit.get("items", []))):
                if item not in ITEM_COMPONENTS_OR_PLACEHOLDERS:
                    holder_rows.append({**base, "unit": name, "item": readable_name(item)})
        for item in set(map(str, items)):
            if item not in ITEM_COMPONENTS_OR_PLACEHOLDERS:
                item_rows.append({**base, "item": readable_name(item)})
        active = []
        for trait in traits:
            tier = int(trait.get("tier_current", trait.get("style", 0)) or 0)
            if tier <= 0:
                continue
            count = int(trait.get("num_units", 0) or 0); name = readable_name(trait["name"])
            signature = f"{name} · tier {tier} · {count} units"
            trait_rows.append({**base, "trait": name, "trait_tier": tier,
                               "units_contributing": count, "trait_threshold": signature})
            active.append((tier, name))
        stars = [int(unit.get("stars", 0) or 0) for unit in units]
        structure_rows.append({**base, "level": float(source.level),
                               "composition_size": float(source.composition_size),
                               "unique_units": len({unit["name"] for unit in units}),
                               "mean_star": np.mean(stars) if stars else 0.0,
                               "max_star": max(stars, default=0),
                               "two_star_units": sum(star == 2 for star in stars),
                               "three_star_units": sum(star >= 3 for star in stars),
                               "total_items": len(items), "unique_items": len(set(items)),
                               "active_traits": len(active),
                               "trait_strength_total": sum(tier for tier, _ in active)})
    return (pd.DataFrame(unit_rows), pd.DataFrame(item_rows), pd.DataFrame(holder_rows),
            pd.DataFrame(trait_rows), pd.DataFrame(structure_rows))


def analyse_associations(participants: pd.DataFrame, units: pd.DataFrame, items: pd.DataFrame,
                         traits: pd.DataFrame) -> dict[str, pd.DataFrame]:
    baseline = float(participants["top4"].mean())
    unit = association_table(units, participants, feature_columns=["unit"],
                             global_rate=baseline, minimum_support=UNIT_MINIMUM)
    unit_star = association_table(units, participants, feature_columns=["unit", "star_level"],
                                  global_rate=baseline, minimum_support=STAR_MINIMUM)
    item = association_table(items, participants, feature_columns=["item"],
                             global_rate=baseline, minimum_support=ITEM_MINIMUM)
    trait = association_table(traits, participants,
                              feature_columns=["trait", "trait_tier", "units_contributing",
                                               "trait_threshold"],
                              global_rate=baseline, minimum_support=TRAIT_MINIMUM)
    for frame in (unit, unit_star, item, trait):
        validate_association_output(frame)
    return {"unit": unit, "unit_star": unit_star, "item": item, "trait": trait}


def interaction_analysis(participants: pd.DataFrame, units: pd.DataFrame, holders: pd.DataFrame,
                         traits: pd.DataFrame, associations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    baseline = float(participants["top4"].mean()); rows = []
    component_rates = {}
    for kind, frame, column in (("unit", associations["unit"], "unit"),
                                ("item", associations["item"], "item"),
                                ("trait", associations["trait"], "trait_threshold")):
        component_rates[kind] = frame.set_index(column)["top_four_rate"].to_dict()

    def add(frame: pd.DataFrame, interaction_type: str, a_col: str, b_col: str,
            a_type: str, b_type: str) -> None:
        for (a, b), group in frame.drop_duplicates(["row_id", a_col, b_col]).groupby([a_col, b_col]):
            n = len(group)
            if n < INTERACTION_MINIMUM:
                continue
            rate = group["top4"].mean(); low, high = wilson_interval(int(group["top4"].sum()), n)
            a_rate = component_rates[a_type].get(a, np.nan)
            b_rate = component_rates[b_type].get(b, np.nan)
            rows.append({"interaction_type": interaction_type, "feature_a": a,
                         "feature_b": b, "appearances": n, "top_four_rate": rate,
                         "top_four_rate_ci95_low": low, "top_four_rate_ci95_high": high,
                         "average_placement": group["placement"].mean(),
                         "uplift_vs_global_pp": 100 * (rate - baseline),
                         "component_a_top_four_rate": a_rate,
                         "uplift_vs_component_a_pp": 100 * (rate - a_rate),
                         "component_b_top_four_rate": b_rate,
                         "uplift_vs_component_b_pp": 100 * (rate - b_rate),
                         "minimum_support": INTERACTION_MINIMUM, "exploratory": True})

    add(holders, "unit_item_holder", "unit", "item", "unit", "item")
    major_units = set(associations["unit"].nlargest(20, "appearances")["unit"])
    major_traits = set(associations["trait"].nlargest(20, "appearances")["trait_threshold"])
    trait_unit = traits.loc[traits["trait_threshold"].isin(major_traits)].merge(
        units.loc[units["unit"].isin(major_units), ["row_id", "unit"]], on="row_id")
    add(trait_unit, "trait_threshold_unit", "trait_threshold", "unit", "trait", "unit")
    selected_traits = traits.loc[traits["trait_threshold"].isin(major_traits),
                                 ["row_id", "trait_threshold", "top4", "placement"]]
    pairs = selected_traits.merge(selected_traits, on="row_id", suffixes=("_a", "_b"))
    pairs = pairs.loc[pairs["trait_threshold_a"] < pairs["trait_threshold_b"]].rename(
        columns={"top4_a": "top4", "placement_a": "placement"})
    add(pairs, "trait_threshold_pair", "trait_threshold_a", "trait_threshold_b", "trait", "trait")
    result = pd.DataFrame(rows).sort_values(
        ["interaction_type", "appearances", "uplift_vs_global_pp"],
        ascending=[True, False, False],
    )
    # Retain a deliberately limited, supported set rather than present an
    # exhaustive combinatorial search as confirmatory evidence.
    return result.groupby("interaction_type", group_keys=False).head(50).reset_index(drop=True)


def neighbouring_trait_comparisons(traits: pd.DataFrame,
                                   trait_analysis: pd.DataFrame) -> pd.DataFrame:
    supported = trait_analysis.loc[trait_analysis["meets_minimum_support"]].copy()
    rows = []
    for trait, group in supported.groupby("trait"):
        ordered = group.sort_values(["units_contributing", "trait_tier"])
        for (_, lower), (_, upper) in zip(ordered.iloc[:-1].iterrows(),
                                          ordered.iloc[1:].iterrows(), strict=False):
            rows.append({"trait": trait, "lower_signature": lower["trait_threshold"],
                         "upper_signature": upper["trait_threshold"],
                         "lower_appearances": lower["appearances"],
                         "upper_appearances": upper["appearances"],
                         "lower_top_four_rate": lower["top_four_rate"],
                         "upper_top_four_rate": upper["top_four_rate"],
                         "difference_upper_minus_lower_pp": 100 * (
                             upper["top_four_rate"] - lower["top_four_rate"]),
                         "descriptive_only": True})
    return pd.DataFrame(rows)


def structure_analysis(structure: pd.DataFrame, global_rate: float) -> pd.DataFrame:
    variables = ["level", "composition_size", "unique_units", "mean_star", "max_star",
                 "two_star_units", "three_star_units", "total_items", "unique_items",
                 "active_traits", "trait_strength_total"]
    rows = []
    for variable in variables:
        values = structure[variable]
        try:
            category = pd.qcut(values, q=5, duplicates="drop")
        except ValueError:
            category = values.astype(str)
        grouped = structure.assign(value_band=category.astype(str)).groupby("value_band", observed=True)
        for band, group in grouped:
            top = int(group["top4"].sum()); n = len(group); low, high = wilson_interval(top, n)
            rows.append({"structure_variable": variable, "value_band": band,
                         "appearances": n, "top_four_rate": top / n,
                         "top_four_rate_ci95_low": low, "top_four_rate_ci95_high": high,
                         "average_placement": group["placement"].mean(),
                         "top_four_uplift_pp": 100 * (top / n - global_rate)})
    return pd.DataFrame(rows)


def rank_analysis(participants: pd.DataFrame, units: pd.DataFrame, items: pd.DataFrame,
                  traits: pd.DataFrame, associations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    baseline = participants["top4"].mean(); rows = []
    selections = [
        ("unit", units, "unit", set(associations["unit"].nlargest(20, "appearances")["unit"])),
        ("item", items, "item", set(associations["item"].nlargest(20, "appearances")["item"])),
        ("trait_threshold", traits, "trait_threshold",
         set(associations["trait"].nlargest(20, "appearances")["trait_threshold"])),
    ]
    for feature_type, frame, column, chosen in selections:
        selected = frame.loc[frame[column].isin(chosen) & frame["rank_tier"].notna()].drop_duplicates(
            ["row_id", column]
        )
        overall = selected.groupby(column)["top4"].mean().to_dict()
        for (feature, rank), group in selected.groupby([column, "rank_tier"]):
            n = len(group); top = int(group["top4"].sum()); low, high = wilson_interval(top, n)
            rows.append({"feature_type": feature_type, "feature": feature, "rank_tier": rank,
                         "appearances": n, "top_four_rate": top / n,
                         "top_four_rate_ci95_low": low, "top_four_rate_ci95_high": high,
                         "average_placement": group["placement"].mean(),
                         "overall_feature_top_four_rate": overall[feature],
                         "rank_difference_pp": 100 * (top / n - overall[feature]),
                         "uplift_vs_global_pp": 100 * (top / n - baseline),
                         "minimum_support": RANK_MINIMUM,
                         "meets_minimum_support": n >= RANK_MINIMUM})
    return pd.DataFrame(rows).sort_values(["feature_type", "feature", "appearances"],
                                          ascending=[True, True, False])


def adjusted_regression(participants: pd.DataFrame, units: pd.DataFrame, items: pd.DataFrame,
                        traits: pd.DataFrame, structure: pd.DataFrame) -> pd.DataFrame:
    development = participants.loc[participants["split"] == "train"].copy().reset_index(drop=True)
    ids = set(development["row_id"]); base = structure.loc[structure["row_id"].isin(ids)].copy()
    base = base.set_index("row_id").reindex(development["row_id"])
    numeric = ["level", "composition_size", "mean_star", "three_star_units", "total_items",
               "active_traits", "trait_strength_total"]
    x = pd.DataFrame(index=development["row_id"])
    labels: dict[str, tuple[str, str]] = {}
    for column in numeric:
        standard = (base[column].to_numpy() - base[column].mean()) / base[column].std(ddof=0)
        name = f"context__{column}"; x[name] = standard
        labels[name] = (column.replace("_", " ").title() + " (per SD)", "context")
    rank = development["rank_tier"].where(development["rank_status"].eq("found"), "UNAVAILABLE")
    rank_dummies = pd.get_dummies(rank, prefix="rank", dtype=float)
    reference = "rank_GOLD" if "rank_GOLD" in rank_dummies else sorted(rank_dummies)[0]
    rank_dummies = rank_dummies.drop(columns=reference)
    rank_dummies.index = development["row_id"]
    x = pd.concat([x, rank_dummies], axis=1)
    for column in rank_dummies:
        labels[column] = (column.removeprefix("rank_").title() + " rank", "rank_context")
    for kind, frame, column, count in (("unit", units, "unit", 10), ("item", items, "item", 10),
                                       ("trait", traits.loc[~traits["trait_threshold"].str.contains("Unique", case=False)],
                                        "trait_threshold", 10)):
        source = frame.loc[frame["row_id"].isin(ids)].drop_duplicates(["row_id", column])
        common = source[column].value_counts().head(count).index
        for feature in common:
            safe = hashlib.sha256(str(feature).encode()).hexdigest()[:12]
            name = f"{kind}__{safe}"; present = set(source.loc[source[column] == feature, "row_id"])
            x[name] = [float(row_id in present) for row_id in x.index]
            labels[name] = (str(feature), kind)
    # Remove near-exact redundant columns before unpenalised inference. This is
    # deterministic and uses the training partition only.
    correlation = x.corr().abs()
    to_drop: set[str] = set()
    for right_index, right in enumerate(x.columns):
        for left in x.columns[:right_index]:
            if left not in to_drop and correlation.loc[left, right] >= .95:
                to_drop.add(right); break
    x = x.drop(columns=sorted(to_drop))
    y = development.set_index("row_id").loc[x.index, "top4"].astype(int).to_numpy()
    model = LogisticRegression(penalty=None, max_iter=3000, solver="lbfgs", random_state=SEED)
    model.fit(x, y)
    design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
    beta = np.r_[model.intercept_[0], model.coef_[0]]
    probability = expit(design @ beta); weights = probability * (1 - probability)
    bread = np.linalg.pinv(design.T @ (design * weights[:, None]))
    scores = design * (y - probability)[:, None]
    groups = development.set_index("row_id").loc[x.index, "match_hash"].astype(str).to_numpy()
    cluster_scores = pd.DataFrame(scores).assign(group=groups).groupby("group").sum().to_numpy()
    meat = cluster_scores.T @ cluster_scores
    covariance = bread @ meat @ bread
    clusters = len(np.unique(groups)); parameters = design.shape[1]
    covariance *= (clusters / (clusters - 1)) * ((len(x) - 1) / (len(x) - parameters))
    se = np.sqrt(np.maximum(np.diag(covariance), 0)); z = beta / se
    p = 2 * stats.norm.sf(np.abs(z)); names = ["intercept", *x.columns]
    result_rows = []
    for index, name in enumerate(names):
        label, family = (("Intercept", "intercept") if name == "intercept" else labels[name])
        result_rows.append({"term": label, "feature_family": family,
                            "coefficient_log_odds": beta[index], "cluster_robust_se": se[index],
                            "odds_ratio": np.exp(beta[index]),
                            "odds_ratio_ci95_low": np.exp(beta[index] - 1.96 * se[index]),
                            "odds_ratio_ci95_high": np.exp(beta[index] + 1.96 * se[index]),
                            "p_value": p[index], "training_participants": len(x),
                            "match_clusters": clusters, "reference_rank": reference.removeprefix("rank_"),
                            "near_exact_correlation_threshold": .95,
                            "near_exact_redundant_terms_removed": len(to_drop),
                            "analysis_scope": "frozen_train_only_associational"})
    result = pd.DataFrame(result_rows)
    result["fdr_bh_q_value"] = benjamini_hochberg(result["p_value"])
    return result


def shap_analysis(root: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = joblib.load(root / "modelling" / "models" / "development" / "top4_f5_xgboost.joblib")
    x = pd.read_csv(root / "data" / "features" / "F5" / "validation_X.csv.gz")
    values = model.get_booster().predict(xgboost.DMatrix(x, feature_names=list(x.columns)),
                                         pred_contribs=True)[:, :-1]
    vocabulary = json.loads((root / "data" / "feature_vocabulary.json").read_text(encoding="utf-8"))
    readable = {}
    for kind, key in (("unit", "unit_vocabulary"), ("trait", "trait_vocabulary"),
                      ("item", "item_vocabulary")):
        for raw in vocabulary[key]:
            readable[(kind, safe_feature_name(raw))] = readable_name(raw)
    rows = []
    for index, feature in enumerate(x.columns):
        present = x.iloc[:, index].to_numpy() != 0
        mean_present = float(values[present, index].mean()) if present.any() else np.nan
        rho = stats.spearmanr(x.iloc[:, index].to_numpy(), values[:, index]).statistic
        rows.append({"feature": readable_feature(feature, readable), "feature_family": feature_group(feature),
                     "mean_absolute_shap": np.abs(values[:, index]).mean(),
                     "mean_signed_shap": values[:, index].mean(),
                     "mean_shap_when_nonzero": mean_present,
                     "feature_value_shap_spearman": rho,
                     "nonzero_validation_observations": int(present.sum()),
                     "higher_value_direction": (
                         "higher_values_top_four_positive" if rho > 0 else
                         "higher_values_bottom_four_negative" if rho < 0 else "undefined"
                     )})
    importance = pd.DataFrame(rows).sort_values("mean_absolute_shap", ascending=False)
    groups = importance.groupby("feature_family", as_index=False).agg(
        encoded_features=("feature", "count"),
        summed_mean_absolute_shap=("mean_absolute_shap", "sum"),
        mean_feature_absolute_shap=("mean_absolute_shap", "mean"),
    ).sort_values("summed_mean_absolute_shap", ascending=False)
    top_indices = [x.columns.get_loc(name) for name in x.columns
                   if readable_feature(name, readable) in set(importance.head(20)["feature"])]
    rng = np.random.default_rng(SEED); sample = rng.choice(len(x), size=min(1500, len(x)), replace=False)
    fig, ax = plt.subplots(figsize=(11, 9))
    for position, feature_index in enumerate(reversed(top_indices)):
        feature_values = x.iloc[sample, feature_index].to_numpy(dtype=float)
        scaled = (feature_values - np.nanmin(feature_values)) / (np.nanmax(feature_values) - np.nanmin(feature_values) + 1e-12)
        jitter = rng.normal(0, .09, len(sample))
        ax.scatter(values[sample, feature_index], position + jitter, c=scaled, cmap="coolwarm",
                   s=8, alpha=.45, edgecolors="none")
    labels = [readable_feature(x.columns[i], readable) for i in reversed(top_indices)]
    ax.set_yticks(range(len(labels)), labels); ax.axvline(0, color="grey", linewidth=.8)
    ax.set_xlabel("TreeSHAP contribution to top-four log odds")
    ax.set_title("Selected XGBoost validation TreeSHAP summary")
    fig.text(.5, .01, "Blue = lower feature value; red = higher feature value. Fixed seed sample.", ha="center")
    fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(output / "figures" / "shap_summary_beeswarm.png", dpi=220); plt.close(fig)
    return importance, groups


def readable_feature(feature: str, vocabulary: dict[tuple[str, str], str] | None = None) -> str:
    replacements = (("unit_presence__", "Unit present · "), ("unit_star__", "Unit star · "),
                    ("trait_presence__", "Trait present · "), ("trait_strength__", "Trait strength · "),
                    ("item_count__", "Item count · "), ("structure__", "Structure · "))
    for prefix, label in replacements:
        if feature.startswith(prefix):
            suffix = feature.removeprefix(prefix)
            kind = "unit" if prefix.startswith("unit_") else (
                "trait" if prefix.startswith("trait_") else "item" if prefix.startswith("item_") else ""
            )
            resolved = (vocabulary or {}).get((kind, suffix), readable_name(suffix))
            return label + resolved
    return readable_name(feature)


def family_interpretation() -> pd.DataFrame:
    path = PROJECT_ROOT / "artifacts" / "thesis_extension_v1" / "tables" / "feature_family_validation_comparison.csv"
    frame = pd.read_csv(path).sort_values("feature_family").reset_index(drop=True)
    frame["feature_added"] = ["basic board state", "units and star levels", "traits",
                              "items", "composition structure"]
    frame["incremental_validation_f1"] = frame["f1"].diff()
    frame["source"] = "existing frozen logistic feature-family validation experiment"
    return frame


def _bar_with_ci(frame: pd.DataFrame, label: str, output: Path, title: str) -> None:
    selected = frame.loc[frame["meets_minimum_support"]].nlargest(15, "top_four_uplift_pp").sort_values("top_four_uplift_pp")
    lower = 100 * (selected["top_four_rate"] - selected["top_four_rate_ci95_low"])
    upper = 100 * (selected["top_four_rate_ci95_high"] - selected["top_four_rate"])
    fig, ax = plt.subplots(figsize=(10, 8)); ax.barh(selected[label], selected["top_four_uplift_pp"], color="#315b7d")
    ax.errorbar(selected["top_four_uplift_pp"], range(len(selected)), xerr=[lower, upper], fmt="none", color="black", capsize=3)
    ax.axvline(0, color="grey", linewidth=.8); ax.set_xlabel("Top-four uplift versus 50% baseline (percentage points)")
    ax.set_title(title); fig.text(.5, .01, f"Minimum support: {int(selected.minimum_support.iloc[0]) if len(selected) else 0} participant boards", ha="center")
    fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(output, dpi=220); plt.close(fig)


def figures(associations: dict[str, pd.DataFrame], rank: pd.DataFrame,
            families: pd.DataFrame, output: Path) -> None:
    _bar_with_ci(associations["unit"], "unit", output / "figures" / "top_units_uplift.png",
                 "Top 15 units by top-four uplift")
    _bar_with_ci(associations["item"], "item", output / "figures" / "top_items_uplift.png",
                 "Top 15 items by top-four uplift")
    traits = associations["trait"].loc[associations["trait"]["meets_minimum_support"]].nlargest(20, "appearances").sort_values("top_four_rate")
    fig, ax = plt.subplots(figsize=(11, 9)); ax.barh(traits["trait_threshold"], traits["top_four_rate"], color="#5b8e7d")
    ax.axvline(.5, color="grey", linestyle="--"); ax.set_xlabel("Top-four rate"); ax.set_title("Supported trait activation signatures")
    fig.tight_layout(); fig.savefig(output / "figures" / "trait_threshold_performance.png", dpi=220); plt.close(fig)
    stars = associations["unit_star"].loc[associations["unit_star"]["meets_minimum_support"]]
    major = associations["unit"].nlargest(10, "appearances")["unit"]
    pivot = stars.loc[stars["unit"].isin(major)].pivot(index="unit", columns="star_level", values="top_four_rate")
    fig, ax = plt.subplots(figsize=(10, 7)); pivot.plot(kind="barh", ax=ax); ax.axvline(.5, color="grey", linestyle="--")
    ax.set(xlabel="Top-four rate", ylabel="Unit", title="Unit performance by star level"); ax.legend(title="Star level")
    fig.tight_layout(); fig.savefig(output / "figures" / "unit_star_comparison.png", dpi=220); plt.close(fig)
    family_plot = families.dropna(subset=["incremental_validation_f1"]).copy()
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(family_plot["feature_family"], family_plot["incremental_validation_f1"], color="#6f8798")
    ax.axhline(0, color="grey", linewidth=.8); ax.set(ylabel="Incremental validation F1", xlabel="Cumulative family",
                                                       title="Added predictive value by cumulative feature family")
    fig.tight_layout(); fig.savefig(output / "figures" / "feature_family_contribution.png", dpi=220); plt.close(fig)
    supported = rank.loc[rank["meets_minimum_support"] & rank["rank_tier"].isin(RANK_ORDER[:5])]
    chosen = supported.groupby("feature")["appearances"].sum().nlargest(12).index
    heat = supported.loc[supported["feature"].isin(chosen)].pivot(index="feature", columns="rank_tier", values="top_four_rate").reindex(columns=RANK_ORDER[:5])
    fig, ax = plt.subplots(figsize=(9, 8)); image = ax.imshow(heat, vmin=.3, vmax=.7, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=30, ha="right"); ax.set_yticks(range(len(heat)), heat.index)
    ax.set_title("Top-four rate for major features by rank tier"); fig.colorbar(image, ax=ax, label="Top-four rate")
    fig.tight_layout(); fig.savefig(output / "figures" / "rank_tier_feature_heatmap.png", dpi=220); plt.close(fig)


def write_reports(output: Path, participants: pd.DataFrame, associations: dict[str, pd.DataFrame],
                  interactions: pd.DataFrame, rank: pd.DataFrame, regression: pd.DataFrame,
                  shap: pd.DataFrame, families: pd.DataFrame) -> None:
    supported_units = associations["unit"].loc[associations["unit"]["meets_minimum_support"]]
    supported_items = associations["item"].loc[associations["item"]["meets_minimum_support"]]
    supported_traits = associations["trait"].loc[associations["trait"]["meets_minimum_support"]]
    top_unit = supported_units.nlargest(1, "top_four_uplift_pp").iloc[0]
    bottom_unit = supported_units.nsmallest(1, "top_four_uplift_pp").iloc[0]
    top_item = supported_items.nlargest(1, "top_four_uplift_pp").iloc[0]
    top_trait = supported_traits.nlargest(1, "top_four_uplift_pp").iloc[0]
    top_interaction = interactions.nlargest(1, "uplift_vs_global_pp").iloc[0]
    summary = f"""# Composition analysis summary

## Frozen-cohort validation

- 40,000 participant end-state observations from 5,000 complete matches.
- Top-four baseline: {participants['top4'].mean():.1%}; average placement: {participants['placement'].mean():.3f}.
- Every match remains wholly within one frozen split. No participant or match hashes are exported.
- Available feature types: units, unit star levels, equipped items, traits with observed activation tier and contributing-unit count, retrieval-time rank tier, player level, and composition-structure summaries.
- Augments are absent from the frozen participant schema and were not analysed.

## Main descriptive findings

Among units with at least {UNIT_MINIMUM} appearances, **{top_unit['unit']}** had the largest positive uplift
({top_unit['top_four_rate']:.1%}; {top_unit['top_four_uplift_pp']:+.1f} pp), while
**{bottom_unit['unit']}** had the strongest negative association
({bottom_unit['top_four_rate']:.1%}; {bottom_unit['top_four_uplift_pp']:+.1f} pp).
The strongest supported item was **{top_item['item']}** ({top_item['top_four_rate']:.1%};
{top_item['top_four_uplift_pp']:+.1f} pp). The strongest supported observed trait signature was
**{top_trait['trait_threshold']}** ({top_trait['top_four_rate']:.1%};
{top_trait['top_four_uplift_pp']:+.1f} pp).

The highest-uplift supported exploratory interaction was **{top_interaction['feature_a']} +
{top_interaction['feature_b']}** ({top_interaction['top_four_rate']:.1%}, n={int(top_interaction['appearances'])}).
Interaction rankings are exploratory and are reported only above {INTERACTION_MINIMUM} observations.

## Model interpretation and feature families

The largest mean-absolute TreeSHAP feature was **{shap.iloc[0]['feature']}**. TreeSHAP explains the
frozen development XGBoost's validation predictions; it does not estimate causal effectiveness.
Cumulative frozen logistic validation F1 increased most when items were added to F3
({families.loc[families.feature_family.eq('F4'), 'incremental_validation_f1'].iloc[0]:+.3f});
F5 added only {families.loc[families.feature_family.eq('F5'), 'incremental_validation_f1'].iloc[0]:+.3f} over F4.

## Inference and limitations

Wilson intervals, exposure odds ratios and Benjamini–Hochberg q-values accompany large-scale
descriptive comparisons. The adjusted regression uses the frozen training partition only and
match-cluster-robust standard errors. These remain observational end-state associations.
Expensive units, high star levels, late-game items and high player level are subject to survivorship
and endogeneity. Rank is a retrieval-time snapshot, not historical match-time rank. Patch-specific
balance and rank subgroup sizes limit generalisation.
"""
    write_text(output / "composition_analysis_summary.md", summary)
    results = f"""# 4.X Composition features associated with top-four placement

## 4.X.1 Unit-level patterns

The frozen cohort contained {len(participants):,} participant boards and had an exact top-four rate
of 50%. Among units satisfying the prespecified {UNIT_MINIMUM}-appearance threshold,
{top_unit['unit']} showed the largest positive top-four uplift ({top_unit['top_four_uplift_pp']:+.1f}
percentage points), whereas {bottom_unit['unit']} showed the largest negative uplift
({bottom_unit['top_four_uplift_pp']:+.1f} percentage points). Estimates for less common units remain
in the machine-readable table but are flagged and excluded from supported rankings.

## 4.X.2 Item-level patterns

At board level, {top_item['item']} was most positively associated with top-four placement among
items meeting the {ITEM_MINIMUM}-board threshold. Holder-conditional unit–item results additionally
compare each combination with the unit's and item's standalone rate, reducing—but not removing—the
confounding created when items are concentrated on stronger holders.

## 4.X.3 Trait-level patterns

Traits were evaluated using observed activation signatures combining `tier_current` and the number
of contributing units. {top_trait['trait_threshold']} had the highest supported uplift. These are
observed payload signatures rather than inferred canonical breakpoint labels.

## 4.X.4 Star-level and composition interactions

Unit–star estimates showed systematically different end-state rates across upgrade levels for many
well-supported units. Limited unit–item, trait–unit and trait–trait combinations were retained only
when support reached {INTERACTION_MINIMUM}. The highest supported interaction was
{top_interaction['feature_a']} with {top_interaction['feature_b']}; all interaction results are exploratory.

## 4.X.5 Rank-tier differences

Rank-stratified estimates were retained for the project's observed tiers and flagged below
{RANK_MINIMUM} observations per cell. Variation across tiers describes heterogeneous association
patterns and should not be interpreted as evidence that rank modifies a causal treatment effect.

## 4.X.6 Adjusted feature associations

The adjusted training-partition logistic model included standardized basic/structure controls,
retrieval-time rank indicators, and a limited set of common units, traits and items. Odds ratios use
match-cluster-robust confidence intervals. Adjustment did not convert the observational end-state
design into a causal analysis; survivorship and endogenous board formation remain central limitations.

## 4.X.7 Model-based feature importance

Native TreeSHAP values from the frozen development XGBoost were calculated on validation data.
{shap.iloc[0]['feature']} had the largest mean absolute attribution. Positive and negative SHAP
values describe contributions to the model's top-four log odds, not effects on match outcomes.
"""
    write_text(output / "thesis_composition_results_draft.md", results)
    discussion = """# Thesis composition discussion points

- Strong units may partly indicate access to late-game economy rather than independent unit strength.
- High-cost and upgraded units are especially vulnerable to survivorship bias.
- Item associations can be confounded by holder selection; conditional holder analyses reduce but do not eliminate this.
- End-state boards are consequences of earlier decisions, resources and survival, not controlled interventions.
- Association and SHAP attribution do not imply causal effectiveness.
- Patch 17.8 balance limits generalisation to later patches and sets.
- Rank-tier heterogeneity may reflect execution, metagame, sampling or snapshot timing.
- Retrieval-time rank may incorporate play after the observed match.
- Multiple comparisons, sparse features and correlated predictors can inflate apparent rankings or divide importance.
- Trait signatures use observed payload tier and unit-count fields; canonical breakpoint semantics were not inferred.
"""
    write_text(output / "thesis_composition_discussion_points.md", discussion)


def ensure_safe_exports(output: Path) -> None:
    for path in output.rglob("*.csv"):
        columns = {str(column).lower() for column in pd.read_csv(path, nrows=0).columns}
        if columns & PROHIBITED:
            raise RuntimeError(f"Prohibited identifier column exported: {path}")


def run(output: Path = OUTPUT_ROOT, source_repository: Path | None = None) -> Path:
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"Versioned output already exists and will not be overwritten: {output}")
    root = locate_production(source_repository); before = production_hashes(root)
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("tables", "figures", "logs"): (output / directory).mkdir()
    participants = load_participants(root)
    units, items, holders, traits, structure = explode_features(participants)
    associations = analyse_associations(participants, units, items, traits)
    interactions = interaction_analysis(participants, units, holders, traits, associations)
    trait_comparisons = neighbouring_trait_comparisons(traits, associations["trait"])
    structure_results = structure_analysis(structure, participants["top4"].mean())
    rank = rank_analysis(participants, units, items, traits, associations)
    regression = adjusted_regression(participants, units, items, traits, structure)
    shap, shap_groups = shap_analysis(root, output)
    families = family_interpretation()

    tables = {
        "unit_top4_analysis.csv": associations["unit"],
        "unit_star_top4_analysis.csv": associations["unit_star"],
        "item_top4_analysis.csv": associations["item"],
        "trait_top4_analysis.csv": associations["trait"],
        "feature_interactions.csv": interactions,
        "rank_stratified_feature_analysis.csv": rank,
        "adjusted_logistic_regression.csv": regression,
        "shap_feature_importance.csv": shap,
        "shap_directional_features.csv": pd.concat([
            shap.loc[shap["feature_value_shap_spearman"] > 0].nlargest(15, "feature_value_shap_spearman").assign(direction="positive"),
            shap.loc[shap["feature_value_shap_spearman"] < 0].nsmallest(15, "feature_value_shap_spearman").assign(direction="negative"),
        ], ignore_index=True),
        "shap_feature_family_importance.csv": shap_groups,
        "feature_family_interpretation.csv": families,
        "composition_structure_analysis.csv": structure_results,
        "trait_threshold_comparisons.csv": trait_comparisons,
        "unit_item_holder_analysis.csv": interactions.loc[interactions["interaction_type"] == "unit_item_holder"],
    }
    empty_augments = pd.DataFrame(columns=["augment", "appearances", "top_four_rate",
                                           "average_placement", "top_four_uplift_pp",
                                           "skip_reason"])
    tables["augment_top4_analysis.csv"] = empty_augments
    for name, frame in tables.items(): frame.to_csv(output / "tables" / name, index=False)
    baseline = pd.DataFrame([{"participant_observations": len(participants),
                              "unique_matches": participants["match_hash"].nunique(),
                              "top_four_rate": participants["top4"].mean(),
                              "average_placement": participants["placement"].mean(),
                              "target_definition": "top4 = placement <= 4",
                              "augment_status": "unavailable_not_fabricated"}])
    baseline.to_csv(output / "tables" / "global_baseline.csv", index=False)
    figures(associations, rank, families, output)
    write_reports(output, participants, associations, interactions, rank, regression, shap, families)
    ensure_safe_exports(output)
    after = production_hashes(root)
    generated = []
    for path in sorted(output.rglob("*")):
        if path.is_file(): generated.append({"path": path.relative_to(output).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest = {"composition_analysis_version": VERSION, "created_at_utc": utc_now(),
                "status": "PASS" if before == after else "FAIL", "mode": "read_only_post_modelling",
                "source_production_root": str(root), "source_analysis_run_id": RUN_ID,
                "dataset_version": "tft_set17_patch17_8_v11", "analysis_contract": "1.0.4",
                "split_sha256": "da64ca53fb584738b9db85c5364fd95644d399fe8202f9ad5b6619f271a95242",
                "cohort": {"participants": len(participants), "matches": participants["match_hash"].nunique(),
                           "top_four_rate": participants["top4"].mean()},
                "thresholds": {"units": UNIT_MINIMUM, "items": ITEM_MINIMUM,
                               "traits": TRAIT_MINIMUM, "unit_star": STAR_MINIMUM,
                               "interactions": INTERACTION_MINIMUM, "rank_cells": RANK_MINIMUM},
                "counts": {"units": len(associations["unit"]), "items": len(associations["item"]),
                           "trait_thresholds": len(associations["trait"]),
                           "reported_interactions": len(interactions)},
                "augments": {"available": False, "reason": "No augment field in frozen participant schema"},
                "completed_item_rule": {"excluded_components_and_placeholders": sorted(ITEM_COMPONENTS_OR_PLACEHOLDERS),
                                        "special_completed_and_artifact_items_retained": True},
                "regression": {"status": "completed", "partition": "train_only",
                               "standard_errors": "match_cluster_robust"},
                "shap": {"status": "completed", "model": "frozen development XGBoost",
                         "partition": "validation", "method": "native TreeSHAP pred_contribs"},
                "test_set_tuning": False, "model_retraining": False,
                "production_artifact_hashes_before_and_after_equal": before == after,
                "source_artifact_hashes": before, "software": {"python": platform.python_version(),
                    "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                    "xgboost": xgboost.__version__}, "seed": SEED, "generated_files": generated}
    write_text(output / "composition_analysis_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    if before != after: raise RuntimeError("Frozen production artifacts changed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only top-four composition association analysis")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--source-repository", type=Path)
    args = parser.parse_args()
    try: result = run(args.output_root, args.source_repository)
    except Exception as exc:
        print(f"COMPOSITION ANALYSIS: FAIL\n{exc}", file=sys.stderr); raise
    print("COMPOSITION ANALYSIS: PASS"); print(f"Outputs: {result}")


if __name__ == "__main__": main()

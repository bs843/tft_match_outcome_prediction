"""Integrate descriptive, adjusted and predictive evidence without model fitting.

This module consumes the immutable composition-analysis exports.  The only
participant-level operation is a predeclared train-versus-validation descriptive
stability check; the final test partition is never used by that computation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analysis.composition_analysis.run import (
    explode_features,
    load_participants,
    locate_production,
    production_hashes,
)
from analysis.composition_analysis.statistics import association_table, benjamini_hochberg
from src.config import PROJECT_ROOT


VERSION = "1.0.0"
SOURCE = PROJECT_ROOT / "artifacts" / "post_model" / "composition_analysis_v1"
OUTPUT = PROJECT_ROOT / "artifacts" / "post_model" / "evidence_synthesis"
THRESHOLDS = {
    "unit": [100, 200, 300, 500],
    "item": [100, 200, 300, 500],
    "trait_signature": [100, 200, 300, 500],
    "unit_star": [100, 200, 300, 500],
    "interaction": [100, 150, 250, 500],
}
PRIMARY_THRESHOLDS = {"unit": 200, "item": 200, "trait_signature": 200,
                      "unit_star": 100, "interaction": 150}
PROHIBITED = {"participant_hash", "match_hash", "puuid", "riot_id", "api_key"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(SOURCE / "tables" / name)


def _base_evidence(frame: pd.DataFrame, feature_type: str, feature: pd.Series) -> pd.DataFrame:
    result = pd.DataFrame({
        "feature": feature.astype(str),
        "feature_type": feature_type,
        "support_n": frame["appearances"],
        "top4_rate": frame["top_four_rate"],
        "baseline_top4_rate": .5,
        "uplift_pp": frame.get("top_four_uplift_pp", frame.get("uplift_vs_global_pp")),
        "wilson_ci_low": frame.get("top_four_rate_ci95_low"),
        "wilson_ci_high": frame.get("top_four_rate_ci95_high"),
        "odds_ratio": frame.get("odds_ratio"),
        "odds_ratio_ci_low": frame.get("odds_ratio_ci95_low"),
        "odds_ratio_ci_high": frame.get("odds_ratio_ci95_high"),
        "p_value": frame.get("fisher_p_value"),
        "q_value": frame.get("fdr_bh_q_value"),
    })
    return result


def descriptive_evidence() -> pd.DataFrame:
    unit = read_table("unit_top4_analysis.csv")
    item = read_table("item_top4_analysis.csv")
    trait = read_table("trait_top4_analysis.csv")
    star = read_table("unit_star_top4_analysis.csv")
    frames = [
        _base_evidence(unit, "unit", unit["unit"]),
        _base_evidence(item, "item", item["item"]),
        _base_evidence(trait, "trait_signature", trait["trait_threshold"]),
        _base_evidence(star, "unit_star", star["unit"] + " · " + star["star_level"].astype(str) + " star"),
    ]
    return pd.concat(frames, ignore_index=True)


def adjusted_evidence() -> pd.DataFrame:
    frame = read_table("adjusted_logistic_regression.csv")
    frame = frame.loc[~frame["feature_family"].isin(["intercept", "rank_context"])].copy()
    type_map = {"unit": "unit", "item": "item", "trait": "trait_signature",
                "context": "composition_structure"}
    frame["feature_type"] = frame["feature_family"].map(type_map)
    replacements = {
        "Level (per SD)": "Level",
        "Composition Size (per SD)": "Composition Size",
        "Mean Star (per SD)": "Structure · Mean Star",
        "Three Star Units (per SD)": "Structure · Three Star Units",
        "Total Items (per SD)": "Structure · Total Items",
        "Active Traits (per SD)": "Structure · Active Traits",
        "Trait Strength Total (per SD)": "Structure · Trait Strength Total",
    }
    frame["feature"] = frame["term"].replace(replacements)
    return frame[["feature", "feature_type", "coefficient_log_odds", "odds_ratio",
                  "odds_ratio_ci95_low", "odds_ratio_ci95_high", "p_value",
                  "fdr_bh_q_value"]].rename(columns={
                      "coefficient_log_odds": "adjusted_coefficient",
                      "odds_ratio": "adjusted_odds_ratio",
                      "odds_ratio_ci95_low": "adjusted_ci_low",
                      "odds_ratio_ci95_high": "adjusted_ci_high",
                      "p_value": "adjusted_p_value",
                      "fdr_bh_q_value": "adjusted_q_value",
                  })


def shap_evidence() -> pd.DataFrame:
    frame = read_table("shap_feature_importance.csv").copy()
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        feature = str(row.feature)
        feature_type: str | None = None
        mapped = feature
        if feature.startswith("Unit present · "):
            feature_type, mapped = "unit", feature.removeprefix("Unit present · ")
        elif feature.startswith("Item count · "):
            feature_type, mapped = "item", feature.removeprefix("Item count · ")
        elif feature.startswith("Structure · "):
            feature_type = "composition_structure"
        elif feature in {"Level", "Composition Size"}:
            feature_type = "composition_structure"
        # Trait presence and unit-star model columns are not forced onto richer
        # descriptive signatures because that would be a speculative mapping.
        if feature_type is not None:
            rows.append({"feature": mapped, "feature_type": feature_type,
                         "mean_abs_shap": float(row.mean_absolute_shap)})
    result = pd.DataFrame(rows)
    result["shap_rank"] = result["mean_abs_shap"].rank(method="min", ascending=False)
    return result


def classify(row: pd.Series, shap_cutoff: float) -> str:
    support = row.get("support_n")
    threshold = PRIMARY_THRESHOLDS.get(str(row.get("feature_type")), 1)
    if pd.notna(support) and float(support) < threshold:
        return "Insufficient evidence"
    raw = row.get("uplift_pp")
    adjusted = row.get("adjusted_coefficient")
    shap = row.get("mean_abs_shap")
    if pd.notna(raw) and pd.notna(adjusted) and np.sign(raw) != np.sign(adjusted):
        return "Directionally unstable"
    if pd.notna(raw) and pd.notna(adjusted):
        raw_or = row.get("odds_ratio")
        adjusted_or = row.get("adjusted_odds_ratio")
        if pd.notna(raw_or) and pd.notna(adjusted_or) and raw_or > 0 and adjusted_or > 0:
            denominator = abs(math.log(raw_or))
            attenuation = 1 - abs(math.log(adjusted_or)) / denominator if denominator else np.nan
            if pd.notna(attenuation) and attenuation >= .5:
                return "Attenuated after adjustment"
        if pd.notna(shap) and shap >= shap_cutoff:
            return "Convergent evidence"
        return "Adjusted and descriptive"
    if pd.notna(shap) and shap >= shap_cutoff and (pd.isna(raw) or abs(raw) < 5):
        return "Predictive-only"
    if pd.notna(raw) and abs(raw) >= 5:
        return "Descriptive-only"
    return "Insufficient evidence"


def feature_evidence_table() -> pd.DataFrame:
    raw = descriptive_evidence()
    adjusted = adjusted_evidence()
    shap = shap_evidence()
    result = raw.merge(adjusted, on=["feature", "feature_type"], how="outer")
    result = result.merge(shap, on=["feature", "feature_type"], how="outer")
    result["descriptive_rank"] = result.groupby("feature_type")["uplift_pp"].transform(
        lambda values: values.abs().rank(method="min", ascending=False)
    )
    result["adjusted_rank"] = result.groupby("feature_type")["adjusted_coefficient"].transform(
        lambda values: values.abs().rank(method="min", ascending=False)
    )
    result["direction_consistent"] = np.where(
        result["uplift_pp"].notna() & result["adjusted_coefficient"].notna(),
        np.sign(result["uplift_pp"]) == np.sign(result["adjusted_coefficient"]),
        pd.NA,
    )
    valid = result["mean_abs_shap"].dropna()
    cutoff = float(valid.quantile(.75)) if len(valid) else math.inf
    result["evidence_category"] = result.apply(classify, axis=1, shap_cutoff=cutoff)
    raw_or = pd.to_numeric(result["odds_ratio"], errors="coerce")
    adjusted_or = pd.to_numeric(result["adjusted_odds_ratio"], errors="coerce")
    denominator = np.abs(np.log(raw_or))
    result["association_attenuation_fraction"] = np.where(
        (raw_or > 0) & (adjusted_or > 0) & (denominator > 0),
        1 - np.abs(np.log(adjusted_or)) / denominator,
        np.nan,
    )
    columns = [
        "feature", "feature_type", "support_n", "top4_rate", "baseline_top4_rate",
        "uplift_pp", "wilson_ci_low", "wilson_ci_high", "odds_ratio",
        "odds_ratio_ci_low", "odds_ratio_ci_high", "p_value", "q_value",
        "adjusted_coefficient", "adjusted_odds_ratio", "adjusted_ci_low",
        "adjusted_ci_high", "adjusted_p_value", "adjusted_q_value", "mean_abs_shap",
        "shap_rank", "descriptive_rank", "adjusted_rank", "direction_consistent",
        "association_attenuation_fraction", "evidence_category",
    ]
    return result[columns].sort_values(["feature_type", "descriptive_rank", "shap_rank"], na_position="last")


def threshold_sensitivity(raw: pd.DataFrame,
                          interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for feature_type, thresholds in THRESHOLDS.items():
        if feature_type == "interaction":
            interaction_rows = interactions.copy()
            interaction_rows["feature_type"] = "interaction"
            interaction_rows["feature"] = (interaction_rows["feature_a"] + " + " +
                                                   interaction_rows["feature_b"])
            interaction_rows = interaction_rows.rename(
                columns={"appearances": "support_n", "uplift_vs_global_pp": "uplift_pp"}
            )
            # The source deliberately retained only interactions at n>=150, so a
            # 100 threshold cannot be reconstructed without rerunning the search.
            for threshold in [150, 250, 500]:
                eligible = interaction_rows.loc[interaction_rows.support_n >= threshold].copy()
                eligible["absolute_uplift_rank"] = eligible.uplift_pp.abs().rank(
                    method="min", ascending=False
                )
                eligible["support_threshold"] = threshold
                rows.append(eligible[["feature_type", "feature", "support_n", "top_four_rate",
                                      "uplift_pp", "support_threshold",
                                      "absolute_uplift_rank"]])
            continue
        group = raw.loc[raw["feature_type"] == feature_type].copy()
        for threshold in thresholds:
            eligible = group.loc[group["support_n"] >= threshold].copy()
            eligible["absolute_uplift_rank"] = eligible["uplift_pp"].abs().rank(
                method="min", ascending=False
            )
            eligible["support_threshold"] = threshold
            rows.append(eligible[["feature_type", "feature", "support_n", "top4_rate",
                                  "uplift_pp", "support_threshold", "absolute_uplift_rank"]])
    sensitivity = pd.concat(rows, ignore_index=True)
    stability_rows = []
    for feature_type, thresholds in THRESHOLDS.items():
        if feature_type == "interaction":
            thresholds = [150, 250, 500]
        for left, right in zip(thresholds[:-1], thresholds[1:], strict=False):
            a = sensitivity.loc[(sensitivity.feature_type == feature_type) &
                                (sensitivity.support_threshold == left)]
            b = sensitivity.loc[(sensitivity.feature_type == feature_type) &
                                (sensitivity.support_threshold == right)]
            joined = a.merge(b, on="feature", suffixes=("_a", "_b"))
            rho = joined["absolute_uplift_rank_a"].corr(
                joined["absolute_uplift_rank_b"], method="spearman"
            ) if len(joined) > 1 else np.nan
            top_a10 = set(a.nsmallest(10, "absolute_uplift_rank").feature)
            top_b10 = set(b.nsmallest(10, "absolute_uplift_rank").feature)
            top_a20 = set(a.nsmallest(20, "absolute_uplift_rank").feature)
            top_b20 = set(b.nsmallest(20, "absolute_uplift_rank").feature)
            stability_rows.append({"feature_type": feature_type,
                                   "threshold_pair": f"{left}→{right}",
                                   "common_features": len(joined), "spearman_rho": rho,
                                   "top10_overlap": len(top_a10 & top_b10),
                                   "top20_overlap": len(top_a20 & top_b20)})
    return sensitivity, pd.DataFrame(stability_rows)


def interaction_strength() -> pd.DataFrame:
    frame = read_table("feature_interactions.csv").copy()
    frame["interaction_excess_over_best_component_pp"] = 100 * (
        frame["top_four_rate"] - frame[["component_a_top_four_rate",
                                         "component_b_top_four_rate"]].max(axis=1)
    )
    total_top = 20_000
    total_bottom = 20_000
    p_values = []
    for row in frame.itertuples(index=False):
        exposed_top = int(round(row.top_four_rate * row.appearances))
        exposed_bottom = int(row.appearances - exposed_top)
        p_values.append(stats.fisher_exact([[exposed_top, exposed_bottom],
                                            [total_top - exposed_top,
                                             total_bottom - exposed_bottom]])[1])
    frame["fisher_p_value"] = p_values
    frame["fdr_bh_q_value"] = benjamini_hochberg(np.asarray(p_values))
    frame["interpretation"] = "exploratory descriptive excess association; not synergy"
    return frame


def multiple_testing(raw: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_type, group in raw.groupby("feature_type"):
        p = group["p_value"].dropna()
        q = group["q_value"].dropna()
        rows.append({"feature_type": feature_type, "raw_comparisons": len(group),
                     "p_lt_0_05": int((p < .05).sum()),
                     "q_lt_0_05": int((q < .05).sum()),
                     "q_lt_0_01": int((q < .01).sum())})
    p = interactions["fisher_p_value"]
    q = interactions["fdr_bh_q_value"]
    rows.append({"feature_type": "interaction", "raw_comparisons": len(interactions),
                 "p_lt_0_05": int((p < .05).sum()), "q_lt_0_05": int((q < .05).sum()),
                 "q_lt_0_01": int((q < .01).sum())})
    return pd.DataFrame(rows)


def development_stability() -> pd.DataFrame:
    root = locate_production()
    participants = load_participants(root)
    development = participants.loc[participants["split"].isin(["train", "validation"])].copy()
    units, items, _, traits, _ = explode_features(development)
    sources = [("unit", units, "unit"), ("item", items, "item"),
               ("trait_signature", traits, "trait_threshold")]
    rows = []
    for feature_type, frame, column in sources:
        partitions = {}
        for split in ("train", "validation"):
            cohort = development.loc[development.split == split]
            exposed = frame.loc[frame.split == split]
            partitions[split] = association_table(
                exposed, cohort, feature_columns=[column], global_rate=.5, minimum_support=1
            ).set_index(column)
        common = partitions["train"].index.intersection(partitions["validation"].index)
        for feature in common:
            train = partitions["train"].loc[feature]
            validation = partitions["validation"].loc[feature]
            if min(train.appearances, validation.appearances) < 100:
                continue
            rows.append({"feature_type": feature_type, "feature": feature,
                         "train_support": int(train.appearances),
                         "validation_support": int(validation.appearances),
                         "train_uplift_pp": float(train.top_four_uplift_pp),
                         "validation_uplift_pp": float(validation.top_four_uplift_pp),
                         "direction_same": bool(np.sign(train.top_four_uplift_pp) ==
                                                np.sign(validation.top_four_uplift_pp)),
                         "absolute_difference_pp": abs(float(train.top_four_uplift_pp -
                                                              validation.top_four_uplift_pp))})
    return pd.DataFrame(rows).sort_values(["feature_type", "train_support"], ascending=[True, False])


def family_incremental() -> pd.DataFrame:
    frame = read_table("feature_family_interpretation.csv").sort_values("feature_family")
    rows = []
    for previous, current in zip(frame.iloc[:-1].itertuples(index=False),
                                 frame.iloc[1:].itertuples(index=False), strict=False):
        rows.append({"transition": f"{previous.feature_family} → {current.feature_family}",
                     "added_information": current.feature_added,
                     "validation_f1_before": previous.f1,
                     "validation_f1_after": current.f1,
                     "validation_f1_change": current.f1 - previous.f1,
                     "source": current.source})
    return pd.DataFrame(rows)


def _plots(evidence: pd.DataFrame, stability: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    mapped = evidence.loc[evidence["uplift_pp"].notna() & evidence["mean_abs_shap"].notna()].copy()
    if not mapped.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        for feature_type, group in mapped.groupby("feature_type"):
            ax.scatter(group["uplift_pp"].abs(), group["mean_abs_shap"], label=feature_type,
                       alpha=.75)
        top = mapped.nlargest(8, "mean_abs_shap")
        for row in top.itertuples(index=False):
            ax.annotate(row.feature, (abs(row.uplift_pp), row.mean_abs_shap), fontsize=7)
        ax.set(xlabel="Absolute top-four uplift (percentage points)",
               ylabel="Mean absolute TreeSHAP",
               title="Descriptive association magnitude versus predictive importance")
        ax.legend(frameon=False); fig.tight_layout()
        fig.savefig(output / "shap_vs_descriptive.png", dpi=220); plt.close(fig)
        ranks = mapped.dropna(subset=["descriptive_rank", "shap_rank"])
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(ranks.descriptive_rank, ranks.shap_rank, alpha=.75)
        limit = max(ranks.descriptive_rank.max(), ranks.shap_rank.max())
        ax.plot([1, limit], [1, limit], linestyle="--", color="grey")
        ax.set(xlabel="Absolute descriptive-association rank", ylabel="TreeSHAP rank",
               title="Descriptive and predictive rankings answer different questions")
        fig.tight_layout(); fig.savefig(output / "descriptive_vs_shap_rank.png", dpi=220)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    plot = stability.copy(); labels = plot.feature_type + " " + plot.threshold_pair
    ax.bar(labels, plot.spearman_rho.fillna(0), color="#557a8b")
    ax.set(ylabel="Spearman rank correlation", title="Ranking stability across support thresholds",
           ylim=(0, 1.05)); ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(output / "support_threshold_rank_stability.png", dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 8)); ax.axis("off")
    stages = ["Frozen Patch 17.8\nend-state data", "Descriptive association",
              "Statistical inference", "Adjusted association",
              "Predictive contribution", "Frozen-model TreeSHAP",
              "Robustness / sensitivity", "Thesis interpretation"]
    ys = np.linspace(.93, .08, len(stages))
    for label, y in zip(stages, ys, strict=False):
        ax.text(.5, y, label, ha="center", va="center",
                bbox={"boxstyle": "round,pad=.5", "fc": "#e9f0f2", "ec": "#31566b"})
    for upper, lower in zip(ys[:-1], ys[1:], strict=False):
        ax.annotate("", xy=(.5, lower + .035), xytext=(.5, upper - .035),
                    arrowprops={"arrowstyle": "->", "color": "#31566b"})
    ax.text(.82, .53, "association ≠ causal effect", color="#9b3d33", weight="bold",
            ha="center", bbox={"boxstyle": "round,pad=.4", "fc": "#fff0ed", "ec": "#9b3d33"})
    fig.tight_layout(); fig.savefig(output / "conceptual_evidence_framework.png", dpi=220)
    plt.close(fig)


def _write_docs(evidence: pd.DataFrame, raw_adjusted: pd.DataFrame,
                families: pd.DataFrame, stability: pd.DataFrame,
                testing: pd.DataFrame, dev: pd.DataFrame) -> None:
    rq = """# Versioned research questions — evidence synthesis v1.0.0

These questions translate, but do not replace, the original thesis aims.

1. **RQ1 — Composition associations:** Which end-state units, items, traits and composition structures are most strongly associated with top-four placement in the frozen Patch 17.8 cohort?
2. **RQ2 — Adjustment:** Which observed end-state composition associations remain after adjustment for approved participant and board-state covariates?
3. **RQ3 — Predictive contribution:** How much incremental predictive information is contributed by increasingly detailed end-state feature families?
4. **RQ4 — Model interpretation:** Which end-state features contribute most strongly to the frozen XGBoost model's validation predictions, and how do these compare with descriptive and adjusted associations?
5. **RQ5 — Exact placement:** To what extent can end-state composition features estimate exact final placement from first to eighth?
6. **RQ6 — Robustness:** How stable are the main descriptive conclusions under predeclared support thresholds and development-partition comparisons?
"""
    (OUTPUT.parent / "research_questions.md").write_text(rq, encoding="utf-8")
    limitations = """# Limitations matrix

| Limitation | Likely effect | Direction of potential bias | Mitigation | Remaining uncertainty |
|---|---|---|---|---|
| Survivorship bias | Later survivors reach higher levels, expensive units, completed items and advanced traits | Likely inflates positive end-state associations with top-four placement | Interpret end-state features as associative; adjust for selected board-state summaries | Survival duration is not fully reconstructed |
| Endogeneity | Final boards are products of prior success as well as predictive summaries | Usually inflates success-associated features; direction can be ambiguous for forced transitions | Separate raw, adjusted and predictive evidence | No counterfactual choice effect is identified |
| Item acquisition | Late/high-value items reflect stage, carousel, encounters and unobserved processes | Often inflates positive associations; some loss-streak access may attenuate or reverse them | Report support, intervals, adjusted estimates and holder context | Acquisition pathway is unavailable |
| Star levels | Upgrades proxy economy, reroll success, survival and commitment | Likely inflates positive star-level associations | Report unit-star support and training-only adjustment summaries | Economy and shops offered are unobserved |
| Rank snapshot timing | Retrieval-time rank may differ from rank when the match occurred | Direction ambiguous; subgroup misclassification may attenuate differences | Keep rank contextual and outside the primary matrix | Historical rank is unavailable |
| Patch specificity | Balance is specific to Patch 17.8 / Set 17 | Limits external validity rather than the internal descriptive association | Freeze and label patch identity | Replication on later patches is required |
| Unobserved decisions | Positioning, shops, rerolls, economy paths and offers are missing | Direction ambiguous and permits residual confounding | Restrict claims to observed end-state data | Full decision trajectory cannot be reconstructed |
| TFT stochasticity | Shops, combat targeting and chance affect boards and outcomes | Generally attenuates deterministic relationships but can add irregular bias | Large frozen cohort and uncertainty reporting | A single historical outcome remains noisy |
| Augment absence | Augments cannot be evaluated | Omits a potentially important source of composition and outcome variation | Do not infer or externally merge augments | Their contribution is unknown |

The dashboard describes and predicts final outcomes; it does not estimate what would have happened if a player had chosen a different unit, item or trait during the match.
"""
    (OUTPUT.parent / "limitations_matrix.md").write_text(limitations, encoding="utf-8")
    claim = """# Claim-strength framework

| Evidence | Permitted wording |
|---|---|
| Raw top-four uplift | “associated with a higher/lower top-four rate” |
| Significant adjusted odds ratio | “remained associated after adjustment for selected observed covariates” |
| TreeSHAP importance | “contributed strongly to the frozen model's predictions” |
| Feature-family F1 gain | “added predictive information” |
| Descriptive, adjusted and predictive agreement | “showed consistent evidence across descriptive, adjusted and predictive analyses” |
| Observational data | Never use “caused”, “improved the player's chance”, or “resulted in” |

Statistical significance indicates evidence against a null association; practical relevance is evaluated separately using effect magnitude and support. Residual and unmeasured confounding remain possible after adjustment.
"""
    (OUTPUT.parent / "claim_strength_framework.md").write_text(claim, encoding="utf-8")
    classification = """# Evidence-classification rules

These labels organize interpretation; they are not additional hypothesis tests.

| Category | Deterministic rule |
|---|---|
| Convergent evidence | Raw and adjusted effects have the same sign, and mapped mean absolute TreeSHAP is in the upper quartile of reliably mapped features |
| Descriptive-only | Absolute raw uplift is at least 5 percentage points, but adjusted or meaningful mapped predictive evidence is absent |
| Predictive-only | Mapped TreeSHAP is in the upper quartile, while raw uplift is absent or below 5 percentage points |
| Attenuated after adjustment | Raw and adjusted odds ratios are valid, directions agree, and `1 - abs(log(adjusted OR)) / abs(log(raw OR)) ≥ 0.50` |
| Directionally unstable | Raw uplift and adjusted log-odds coefficient have opposite signs |
| Adjusted and descriptive | Raw and adjusted effects agree, but mapped TreeSHAP is absent or below the upper-quartile threshold |
| Insufficient evidence | Support is below the prespecified primary threshold, or the available evidence does not meet another rule |

The attenuation quantity is **association attenuation after covariate adjustment**.
It must not be described as the proportion of confounding explained. Null fields are
retained whenever two feature representations cannot be mapped without speculation.
"""
    (OUTPUT / "evidence_classification_rules.md").write_text(classification, encoding="utf-8")
    requested = [
        ("unit", "Shen"),
        ("unit", "Aurora"),
        ("item", "Force Of Nature"),
        ("trait_signature", "Space Groove · tier 4 · 7 units"),
    ]
    headline_parts = [raw_adjusted.loc[(raw_adjusted.feature_type == kind) &
                                       (raw_adjusted.feature == feature)]
                      for kind, feature in requested]
    supported_mapped = raw_adjusted.loc[
        raw_adjusted.adjusted_odds_ratio.notna() &
        (raw_adjusted.support_n >= raw_adjusted.feature_type.map(PRIMARY_THRESHOLDS))
    ].sort_values("uplift_pp", key=lambda values: values.abs(), ascending=False).head(6)
    headline = pd.concat([*headline_parts, supported_mapped], ignore_index=True).drop_duplicates(
        ["feature_type", "feature"]
    )
    lines = ["# Raw versus adjusted association summary", "",
             "Adjustment uses the frozen training partition and selected observed covariates with match-cluster-robust standard errors. It does not remove all confounding.", ""]
    for row in headline.itertuples(index=False):
        adjusted = (f"adjusted OR {row.adjusted_odds_ratio:.2f} "
                    f"(95% CI {row.adjusted_ci_low:.2f}–{row.adjusted_ci_high:.2f})"
                    if pd.notna(row.adjusted_odds_ratio) else "no directly mapped adjusted estimate")
        lines.append(f"- **{row.feature}**: {row.top4_rate:.1%} top-four, {row.uplift_pp:+.1f} pp, n={int(row.support_n):,}; raw OR {row.odds_ratio:.2f}; {adjusted}.")
    lines += ["", "Attenuation is labelled association attenuation after covariate adjustment and is not interpreted as confounding explained."]
    (OUTPUT / "raw_vs_adjusted_summary.md").write_text("\n".join(lines), encoding="utf-8")
    best_transition = families.loc[families.validation_f1_change.idxmax()]
    direction_rate = float(dev.direction_same.mean()) if len(dev) else np.nan
    synthesis = f"""# Master's-level results synthesis

## 1. Research-question mapping

| RQ | Analysis | Artifact | Core finding |
|---|---|---|---|
| RQ1 | Support-filtered descriptive association | `feature_evidence_table.csv` | Unit, item, trait-signature and unit-star outcomes vary materially in the frozen cohort. |
| RQ2 | Training-only adjusted logistic synthesis | `raw_vs_adjusted_associations.csv` | Some raw associations persist; others attenuate or change direction after selected-covariate adjustment. |
| RQ3 | Frozen F1–F5 validation experiment | `feature_family_incremental_value.csv` | The largest transition was {best_transition.transition} ({best_transition.validation_f1_change:+.3f} F1). |
| RQ4 | Locked XGBoost validation TreeSHAP | `shap_vs_descriptive.csv` | Predictive importance and raw uplift are related but non-interchangeable quantities. |
| RQ5 | Frozen secondary placement evaluation | `../thesis_extension_v1/tables/placement_final_test_metrics.csv` | Exact placement is better than the dummy baseline but materially harder than top-four classification. |
| RQ6 | Threshold and development-partition sensitivity | `support_threshold_sensitivity.csv` | Ranking and directional stability are reported without using test outcomes for discovery. |

## 2. Composition associations

Shen, Force Of Nature and the Space Groove tier-4 / seven-unit signature were among the strongest supported positive descriptive findings. Aurora was a leading negative unit association. These are final-board associations, not recommended interventions.

## 3. Adjustment

The integrated table distinguishes persisted, attenuated and directionally unstable findings. Adjustment is restricted to the frozen training partition and selected observed covariates; residual endogeneity and survivorship remain.

## 4. Predictive contribution

Units/star levels created the largest cumulative improvement over F1. Items added a further substantial increment at F3 → F4, while F4 → F5 added approximately 0.002 F1. Incremental prediction is not causal importance.

## 5. Model interpretation

TreeSHAP explains the locked XGBoost's validation predictions. Total item count, mean star level and player level were major model contributions. High SHAP importance can coexist with modest raw uplift because the model uses conditional nonlinear patterns.

## 6. Robustness

Predeclared support thresholds were 100, 200, 300 and 500 for main features. The existing interaction artifact begins at n=150, so interaction sensitivity could be evaluated at 150, 250 and 500 but not reconstructed at 100 without rerunning the exploratory search. Across {len(dev):,} supported development comparisons, {direction_rate:.1%} retained the same train/validation direction. BH summaries separate raw p-values from multiplicity-adjusted evidence.

## 7. Exact-placement secondary analysis

The locked random forest achieved MAE 1.065, median absolute error 1.0, 34.9% exact accuracy, 72.4% within one place and 89.9% within two, versus dummy MAE 3.5. No new test evaluation was performed.

## 8. Limitations

Survivorship, endogeneity, item acquisition, upgrade economics, retrieval-time rank, missing decision trajectories, stochasticity and patch specificity constrain interpretation. Augments were absent from the frozen schema and were not reconstructed.

## 9. Overall answer

The project supports a coherent progression from descriptive association, through selected-covariate adjustment, to incremental predictive contribution and frozen-model interpretation. It does not identify causal effects of choosing a unit, item or trait.
"""
    (OUTPUT.parent / "masters_level_results_synthesis.md").write_text(synthesis, encoding="utf-8")
    discussion = """# Thesis Discussion v2 — outline

## 5.1 Summary of principal findings
Tie each finding explicitly to RQ1–RQ6 and distinguish magnitude, uncertainty and support.

## 5.2 Which composition features were associated with top-four placement?
Discuss supported units, completed items, observed trait activation signatures and board structure. Use “associated with”.

## 5.3 Raw versus adjusted associations
Discuss persistence, attenuation and directional instability. State that selected-covariate adjustment does not establish causality.

## 5.4 What information contributed most to prediction?
Interpret the frozen F1–F5 increments, diminishing gains, and the small F4→F5 increment.

## 5.5 Descriptive association versus model importance
Contrast top-four uplift with TreeSHAP and explain disagreements caused by conditional, nonlinear and correlated predictors.

## 5.6 Exact-placement prediction
Explain why eight-class ordered prediction is harder than a balanced binary outcome; report MAE and tolerance accuracies against the dummy.

## 5.7 Comparison with previous literature
[Insert verified TFT / esports analytics citation.] [Insert interpretable machine-learning citation.] Do not invent references.

## 5.8 Robustness and reliability
Discuss predeclared support thresholds, rank stability, BH correction and train/validation directional replication.

## 5.9 Limitations
Use `limitations_matrix.md`, including likely bias direction and remaining uncertainty.

## 5.10 Contribution
The contribution is an integrated, reproducible account of descriptive patterns, adjusted observational relationships, incremental predictive information and locked-model interpretation—not merely an XGBoost classifier.

## 5.11 Future work
Longitudinal rank, multiple patches, prospective validation, compliant earlier-game states, causal/quasi-causal designs and external replication were not performed and remain future work.
"""
    (OUTPUT.parent / "thesis_discussion_v2.md").write_text(discussion, encoding="utf-8")
    mapping = """# Proposal-to-final mapping

| Original objective | Final analysis | Artifact | Status | Deviation / reason |
|---|---|---|---|---|
| Compare top-four and bottom-four composition patterns | Support-filtered unit/item/trait/star/interaction analysis with Wilson intervals, odds ratios and BH correction | `composition_analysis_v1/tables/` | Completed and strengthened | Added inference and robustness |
| Evaluate increasingly detailed feature sets | Frozen cumulative F1–F5 validation comparison | `evidence_synthesis/feature_family_incremental_value.csv` | Completed | None |
| Interpret the selected model | Locked XGBoost validation TreeSHAP plus association comparison | `evidence_synthesis/shap_vs_descriptive.csv` | Completed and strengthened | No model refit |
| Predict exact placement | Locked random-forest secondary evaluation | `thesis_extension_v1/tables/placement_final_test_metrics.csv` | Completed | Secondary task |
| Examine rank | Retrieval-time rank contextual descriptions only | `composition_analysis_v1/tables/rank_stratified_feature_analysis.csv` | Partial / contextual only | Rank is not historical match-time rank and remains outside prediction |
| Analyse augments where available | No augment analysis | `composition_analysis_v1/tables/augment_top4_analysis.csv` | Not analysed | Absent from frozen participant schema; not inferred or externally merged |
"""
    (OUTPUT.parent / "proposal_to_final_mapping.md").write_text(mapping, encoding="utf-8")


def run(output: Path = OUTPUT) -> dict[str, Any]:
    global OUTPUT
    if output != OUTPUT:
        OUTPUT = output
    output.mkdir(parents=True, exist_ok=False)
    (output / "figures").mkdir()
    source_files = sorted(SOURCE.rglob("*"))
    source_hashes_before = {str(path.relative_to(PROJECT_ROOT)): sha256(path)
                            for path in source_files if path.is_file()}
    raw = descriptive_evidence()
    evidence = feature_evidence_table()
    raw_adjusted = evidence.loc[evidence["uplift_pp"].notna()].copy()
    shap_compare = evidence.loc[evidence["uplift_pp"].notna() &
                                evidence["mean_abs_shap"].notna()].copy()
    interactions = interaction_strength()
    sensitivity, stability = threshold_sensitivity(raw, interactions)
    testing = multiple_testing(raw, interactions)
    development = development_stability()
    families = family_incremental()
    tables = {
        "feature_evidence_table.csv": evidence,
        "raw_vs_adjusted_associations.csv": raw_adjusted,
        "shap_vs_descriptive.csv": shap_compare,
        "feature_family_incremental_value.csv": families,
        "support_threshold_sensitivity.csv": sensitivity,
        "threshold_rank_stability.csv": stability,
        "multiple_testing_summary.csv": testing,
        "train_validation_descriptive_stability.csv": development,
        "interaction_strength_comparison.csv": interactions,
    }
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    _plots(evidence, stability, output / "figures")
    _write_docs(evidence, raw_adjusted, families, stability, testing, development)
    robustness = f"""# Robustness summary

- Support thresholds were predeclared: 100/200/300/500 for units, items, traits and unit-star rows; 100/150/250/500 for exploratory interactions.
- Rank correlations and top-10/top-20 overlaps are in `threshold_rank_stability.csv`.
- Multiple-testing counts are in `multiple_testing_summary.csv`; BH q-values are the sole multiplicity correction.
- Train-versus-validation descriptive stability uses only development partitions. No final-test outcome was used for discovery or threshold choice.
- {int(development.direction_same.sum()):,}/{len(development):,} supported development comparisons retained direction.
- Rank remains retrieval-time context only and is absent from the frozen primary model.
"""
    (output / "robustness_summary.md").write_text(robustness, encoding="utf-8")
    source_hashes_after = {str(path.relative_to(PROJECT_ROOT)): sha256(path)
                           for path in source_files if path.is_file()}
    production_root = locate_production()
    production_before = production_hashes(production_root)
    production_after = production_hashes(production_root)
    exported_columns = {column.lower() for frame in tables.values() for column in frame.columns}
    if PROHIBITED & exported_columns:
        raise RuntimeError("Identifier entered an evidence-synthesis export")
    manifest = {
        "evidence_synthesis_version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "analysis_contract": "1.0.4",
        "dataset_version": "tft_set17_patch17_8_v11",
        "participants": 40_000,
        "matches": 5_000,
        "baseline_top4_rate": .5,
        "average_placement": 4.5,
        "source_composition_hashes_unchanged": source_hashes_before == source_hashes_after,
        "production_artifact_hashes_unchanged": production_before == production_after,
        "production_model_retrained": False,
        "model_selection_changed": False,
        "hyperparameters_changed": False,
        "split_changed": False,
        "test_based_tuning": False,
        "test_used_for_new_discovery": False,
        "adjusted_regression_partition": "frozen_train_only_from_authoritative_source",
        "shap_source": "existing frozen development XGBoost validation TreeSHAP",
        "rank_in_primary_matrix": False,
        "augment_analysis": "not_generated_schema_absent",
        "thresholds": THRESHOLDS,
        "generated_files": sorted(str(path.relative_to(output)) for path in output.rglob("*")
                                  if path.is_file()),
    }
    (output / "evidence_synthesis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Master's-level evidence synthesis")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = run(args.output)
    print(f"EVIDENCE SYNTHESIS: {manifest['status']}")
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()

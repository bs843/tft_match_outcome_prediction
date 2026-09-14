from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from analysis.post_model.common import sha256_file, utc_now, write_table, write_text
from analysis.post_model.verify import verify_inputs
from analysis.src.features import CompositionFeatureEncoder
from analysis.src.splits import cv_splits
from analysis.thesis_extension.run import GROUPS, OUTPUT_ROOT, PRODUCTION_RUN_ID, SEED, _xgb


FILES = (
    "tables/cv_participant_overlap_audit.csv",
    "tables/player_connected_component_summary.csv",
    "tables/player_component_cv_folds.csv",
    "tables/player_component_cv_summary.csv",
    "tables/player_component_cv_summary.md",
    "related_entity_sensitivity.md",
    "related_entity_sensitivity_manifest.json",
)


def connected_match_components(frame: pd.DataFrame) -> pd.Series:
    matches = list(frame["match_hash"].astype(str).unique())
    parent = {match: match for match in matches}; size = {match: 1 for match in matches}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]; value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left = find(left); right = find(right)
        if left == right: return
        if size[left] < size[right]: left, right = right, left
        parent[right] = left; size[left] += size[right]

    for _, values in frame.groupby("participant_hash")["match_hash"]:
        group = list(values.astype(str).unique())
        for match in group[1:]: union(group[0], match)
    return frame["match_hash"].astype(str).map(lambda match: find(match))


def run(output: Path = OUTPUT_ROOT) -> Path:
    output = output.resolve()
    if not (output / "thesis_extension_manifest.json").is_file():
        raise RuntimeError("Completed thesis extension is required")
    existing = [str(output / name) for name in FILES if (output / name).exists()]
    if existing: raise RuntimeError("Related-entity outputs will not be overwritten: " + "; ".join(existing))
    verified = verify_inputs(PRODUCTION_RUN_ID, output, require_output_absent=False)
    before = verified.artifact_hashes.copy(); production = verified.production_root
    train = pd.read_csv(production / "data" / "train_participants.csv.gz").reset_index(drop=True)

    overlap_rows = []
    for fold, (fit_idx, held_idx) in enumerate(cv_splits(train, folds=5, seed=SEED), 1):
        fit_players = set(train.iloc[fit_idx]["participant_hash"])
        held_players = set(train.iloc[held_idx]["participant_hash"])
        overlap_rows.append({"fold": fold, "fit_matches": train.iloc[fit_idx]["match_hash"].nunique(),
                             "held_matches": train.iloc[held_idx]["match_hash"].nunique(),
                             "held_unique_participants": len(held_players),
                             "participants_also_in_fit": len(fit_players & held_players),
                             "match_overlap": len(set(train.iloc[fit_idx]["match_hash"]) &
                                                  set(train.iloc[held_idx]["match_hash"]))})
    write_table(pd.DataFrame(overlap_rows), output / "tables" / "cv_participant_overlap_audit.csv")

    train["component"] = connected_match_components(train)
    match_components = train[["match_hash", "component"]].drop_duplicates()
    sizes = match_components.groupby("component").size().sort_values(ascending=False)
    component_summary = pd.DataFrame([{
        "matches": len(match_components), "components": len(sizes),
        "largest_component_matches": int(sizes.max()), "median_component_matches": float(sizes.median()),
        "components_with_multiple_matches": int((sizes > 1).sum()),
        "largest_component_fraction": float(sizes.max() / len(match_components)),
    }])
    write_table(component_summary, output / "tables" / "player_connected_component_summary.csv")

    match_level = match_components.reset_index(drop=True)
    splitter = GroupKFold(n_splits=5)
    rows = []
    for fold, (fit_match_idx, held_match_idx) in enumerate(
        splitter.split(match_level, groups=match_level["component"]), 1
    ):
        fit_matches = set(match_level.iloc[fit_match_idx]["match_hash"])
        held_matches = set(match_level.iloc[held_match_idx]["match_hash"])
        fit = train.loc[train["match_hash"].isin(fit_matches)].drop(columns="component").reset_index(drop=True)
        held = train.loc[train["match_hash"].isin(held_matches)].drop(columns="component").reset_index(drop=True)
        player_overlap = len(set(fit["participant_hash"]) & set(held["participant_hash"]))
        if player_overlap or fit_matches & held_matches:
            raise RuntimeError("Player-component fold has related-entity leakage")
        encoder = CompositionFeatureEncoder(5).fit(fit)
        fit_x = encoder.transform(fit, GROUPS); held_x = encoder.transform(held, GROUPS)
        model = _xgb(SEED).fit(fit_x, fit["top4"].astype(int))
        score = f1_score(held["top4"].astype(int), model.predict(held_x), zero_division=0)
        rows.append({"fold": fold, "fit_matches": len(fit_matches),
                     "held_matches": len(held_matches), "fit_rows": len(fit), "held_rows": len(held),
                     "participant_overlap": player_overlap, "f1": score,
                     "encoded_columns": fit_x.shape[1], "seed": SEED,
                     "splitter": "GroupKFold_player_connected_match_components"})
    folds = pd.DataFrame(rows)
    summary = pd.DataFrame([{"folds": len(folds), "mean_f1": folds["f1"].mean(),
                             "sd_f1": folds["f1"].std(ddof=1), "min_f1": folds["f1"].min(),
                             "max_f1": folds["f1"].max(), "participant_overlap_all_folds": 0,
                             "seed": SEED}])
    write_table(folds, output / "tables" / "player_component_cv_folds.csv")
    write_table(summary, output / "tables" / "player_component_cv_summary.csv",
                output / "tables" / "player_component_cv_summary.md", decimals=4)
    text = f"""# Related-entity sensitivity analysis

The contract CV groups all eight observations from a match, so match overlap is zero. However,
the ordinary five folds contain 1,009–1,052 participant identities that also occur in their
corresponding fitting fold. This can make ordinary CV optimistic if stable player-specific board
patterns are learnable, even though participant identifiers themselves are prohibited features.

Training matches form {int(component_summary.iloc[0]['components'])} player-connected components;
the largest contains {int(component_summary.iloc[0]['largest_component_matches'])} matches
({component_summary.iloc[0]['largest_component_fraction']:.1%} of training matches). A stricter
GroupKFold sensitivity kept every connected component within one fold, achieved zero participant
overlap in all folds, and produced mean F1 {summary.iloc[0]['mean_f1']:.3f}
(fold SD {summary.iloc[0]['sd_f1']:.3f}). Fold sizes vary because connected components are indivisible.
This post-hoc sensitivity does not replace the authoritative chronological split or alter selection.
"""
    write_text(output / "related_entity_sensitivity.md", text)
    after = {path: sha256_file(production / path) for path in before}
    manifest = {"created_at_utc": utc_now(), "status": "PASS" if before == after else "FAIL",
                "grouping": "player_connected_match_components", "folds": 5, "seed": SEED,
                "production_artifact_hashes_before_and_after_equal": before == after,
                "files": {name: sha256_file(output / name) for name in FILES[:-1]}}
    write_text(output / "related_entity_sensitivity_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    if before != after: raise RuntimeError("Frozen production artifacts changed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(); result = run(args.output_root)
    print("RELATED-ENTITY SENSITIVITY: PASS"); print(f"Outputs: {result}")


if __name__ == "__main__": main()

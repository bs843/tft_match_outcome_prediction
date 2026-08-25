from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest
import yaml

import analysis.run as production_run
import analysis.src.contract as contract_module
from analysis.src.contract import (AnalysisContractError, load_contract,
                                   run_analysis_preflight, verify_analysis_release)
from analysis.src.gate import AnalysisApproval
from analysis.src.features import validate_matrix_contract
from analysis.src.models import (PLACEMENT_TASK, TOP4_TASK, _estimators,
                                 run_final_evaluation)
from analysis.src.splits import (FrozenSplitError, canonical_split_hash,
                                 load_frozen_match_split, validate_frozen_assignments)
from src.config import load_config


def rows(counts: dict[str, int], version: str = "dataset") -> list[dict[str, str]]:
    result = []
    number = 0
    for split, count in counts.items():
        for _ in range(count):
            result.append({"match_hash": f"m{number:05d}", "split": split,
                           "dataset_version": version})
            number += 1
    return result


def test_authoritative_split_is_exact_70_15_15_and_deterministic() -> None:
    counts = {"train": 3500, "validation": 750, "test": 750}
    assignments = rows(counts, "tft_set17_patch17_8_v11")
    expected = {row["match_hash"] for row in assignments}
    first = validate_frozen_assignments(
        assignments, expected_match_ids=expected, expected_counts=counts,
        dataset_version="tft_set17_patch17_8_v11",
    )
    second = canonical_split_hash(list(reversed(assignments)))
    assert first == second


def test_split_rejects_overlap_missing_and_unexpected_ids() -> None:
    assignments = rows({"train": 2, "validation": 1, "test": 1})
    assignments[1]["match_hash"] = assignments[0]["match_hash"]
    with pytest.raises(FrozenSplitError, match="duplicate"):
        validate_frozen_assignments(
            assignments, expected_match_ids={"m00000", "m00001", "m00002", "m00003"},
            expected_counts={"train": 2, "validation": 1, "test": 1},
            dataset_version="dataset",
        )


def test_participants_from_one_match_cannot_cross_frozen_partitions(tmp_path: Path) -> None:
    assignments = rows({"train": 1, "validation": 1, "test": 1})
    path = tmp_path / "split.csv"
    pd.DataFrame(assignments).to_csv(path, index=False)
    frame = pd.DataFrame([
        {"match_hash": row["match_hash"], "participant_hash": f"p-{index}-{slot}"}
        for index, row in enumerate(assignments) for slot in range(8)
    ])
    split = load_frozen_match_split(
        frame, path, expected_counts={"train": 1, "validation": 1, "test": 1},
        dataset_version="dataset", expected_hash=canonical_split_hash(assignments),
    )
    assert [part["match_hash"].nunique() for part in
            (split.train, split.validation, split.test)] == [1, 1, 1]
    assert all(len(part) == 8 for part in (split.train, split.validation, split.test))


def test_bound_file_hash_detects_modification(tmp_path: Path) -> None:
    path = tmp_path / "experiments.yaml"
    path.write_text("approved", encoding="utf-8")
    expected = contract_module.sha256_file(path)
    contract_module._verify_hash(path, expected, "experiments.yaml")
    path.write_text("modified", encoding="utf-8")
    with pytest.raises(AnalysisContractError, match="expected.*observed.*new analysis-contract"):
        contract_module._verify_hash(path, expected, "experiments.yaml")


@pytest.mark.parametrize("name", ["experiments.yaml", "columns.yaml", "split.csv"])
def test_each_contract_bound_hash_fails_after_change(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_text("frozen", encoding="utf-8")
    expected = contract_module.sha256_file(path)
    contract_module._verify_hash(path, expected, name)
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(AnalysisContractError, match="hash mismatch"):
        contract_module._verify_hash(path, expected, name)


def test_release_enforcement_requires_clean_exact_annotated_tag(monkeypatch) -> None:
    contract = {"release": {
        "required_analysis_tag": "approved-tag",
        "required_analysis_commit": "resolve_from_required_analysis_tag",
        "require_annotated_tag": True,
    }}
    values = {
        ("status", "--porcelain", "--untracked-files=all"): "",
        ("cat-file", "-t", "refs/tags/approved-tag"): "tag",
        ("rev-parse", "approved-tag^{commit}"): "abc",
        ("rev-parse", "HEAD"): "abc",
    }
    monkeypatch.setattr(contract_module, "_git", lambda *args: values[args])
    assert verify_analysis_release(contract) == ("abc", "approved-tag")
    values[("rev-parse", "HEAD")] = "wrong"
    with pytest.raises(AnalysisContractError, match="HEAD mismatch"):
        verify_analysis_release(contract)


def test_dirty_worktree_fails_release(monkeypatch) -> None:
    contract = {"release": {
        "required_analysis_tag": "approved-tag",
        "required_analysis_commit": "resolve_from_required_analysis_tag",
    }}
    monkeypatch.setattr(contract_module, "_git", lambda *args: "M analysis/run.py")
    with pytest.raises(AnalysisContractError, match="clean Git"):
        verify_analysis_release(contract)


def test_missing_analysis_tag_fails_release(monkeypatch) -> None:
    contract = {"release": {
        "required_analysis_tag": "missing-tag",
        "required_analysis_commit": "resolve_from_required_analysis_tag",
    }}
    def fake_git(*args):
        if args[0] == "status":
            return ""
        raise AnalysisContractError("missing")
    monkeypatch.setattr(contract_module, "_git", fake_git)
    with pytest.raises(AnalysisContractError, match="release tag is missing"):
        verify_analysis_release(contract)


def test_wrong_contract_version_fails(tmp_path: Path) -> None:
    path = tmp_path / "contract.yaml"
    path.write_text('analysis_contract_version: "2.0.0"\n', encoding="utf-8")
    with pytest.raises(AnalysisContractError, match="version"):
        load_contract(path)


def test_feature_contract_accepts_approved_and_rejects_rank_or_missing() -> None:
    contract = load_contract()
    approved = ["level", "composition_size"]
    validate_matrix_contract(approved, family="F1", contract=contract)
    with pytest.raises(RuntimeError, match="Rank snapshot"):
        validate_matrix_contract(approved + ["rank_snapshot_tier"], family="F1",
                                 contract=contract)
    with pytest.raises(RuntimeError, match="missing contracted"):
        validate_matrix_contract(["level"], family="F1", contract=contract)


def test_final_evaluation_cannot_open_before_selection_is_locked(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Locked selection"):
        run_final_evaluation(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}, {}, tmp_path,
            selections={}, seed=4275, access_events={},
        )


def test_final_evaluation_requires_both_task_selections(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="independently for both production tasks"):
        run_final_evaluation(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}, {}, tmp_path,
            selections={TOP4_TASK: {"selected_model": "dummy"}}, seed=4275,
            access_events={PLACEMENT_TASK: {}},
        )


def test_production_path_cannot_invoke_legacy_80_20_split() -> None:
    source = inspect.getsource(production_run.run)
    assert "create_match_split" not in source
    assert "test_fraction" not in source


def test_repository_scientific_contract_is_self_consistent() -> None:
    contract = load_contract()
    experiments_path = contract_module.PROJECT_ROOT / contract["configuration"]["experiments_file"]
    columns_path = contract_module.PROJECT_ROOT / contract["configuration"]["columns_file"]
    experiments = yaml.safe_load(experiments_path.read_text(encoding="utf-8"))
    contract_module._verify_scientific_configuration(contract, experiments)
    contract_module._verify_text_hash(
        experiments_path, contract["configuration"]["experiments_sha256"],
        "experiments.yaml",
    )
    contract_module._verify_text_hash(
        columns_path, contract["configuration"]["columns_sha256"], "columns.yaml",
    )


def test_text_configuration_hash_is_newline_invariant(tmp_path: Path) -> None:
    lf = tmp_path / "lf.yaml"
    crlf = tmp_path / "crlf.yaml"
    lf.write_bytes(b"key: value\nitems:\n  - one\n")
    crlf.write_bytes(b"key: value\r\nitems:\r\n  - one\r\n")
    expected = contract_module.sha256_text_file(lf)
    assert contract_module.sha256_text_file(crlf) == expected
    contract_module._verify_text_hash(crlf, expected, "portable.yaml")


def test_wrong_dataset_version_fails_candidate_preflight() -> None:
    config = load_config()
    config["project"]["dataset_version"] = "wrong"
    with pytest.raises(AnalysisContractError, match="dataset version"):
        run_analysis_preflight(config, verify_release=False)


def test_wrong_collection_provenance_fails_candidate_preflight(monkeypatch) -> None:
    config = load_config()
    approved = AnalysisApproval(
        collection_run_id="20260823T124729Z-673b6e6c",
        collection_manifest=Path("collection.json"),
        validation_summary=Path("validation.json"),
        validated_manifest=Path("validated.csv"),
        validated_files=(), validated_file_map={},
        accepted_count=5000, participant_count=40000,
        dataset_version="tft_set17_patch17_8_v11", schema_version="1.3.2",
        git_commit="collection-commit", git_tag="wrong-tag",
        checksums={"accepted_manifest_content_sha256":
                   "5f3e67c9eac8bc01c395eb2d80ebdbff3113b124c2f0929b92dd1321b43ff581"},
    )
    monkeypatch.setattr(
        contract_module, "require_analysis_approval",
        lambda *_args, **_kwargs: approved,
    )
    with pytest.raises(AnalysisContractError, match="collection provenance tag"):
        run_analysis_preflight(config, verify_release=False)


def test_fixed_seed_baseline_is_reproducible() -> None:
    x = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0] * 4,
                      "b": [1.0, 0.0, 1.0, 0.0] * 4})
    y = pd.Series([0, 1, 0, 1] * 4)
    first = _estimators(4275, require_boosters=False)["random_forest"]
    second = _estimators(4275, require_boosters=False)["random_forest"]
    first.fit(x, y); second.fit(x, y)
    assert (first.predict_proba(x) == second.predict_proba(x)).all()

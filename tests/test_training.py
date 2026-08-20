from __future__ import annotations

from pathlib import Path

from src.train_model import train_and_evaluate
from src.utils import load_json, write_csv_rows


def test_training_writes_frozen_test_metrics(tmp_path: Path) -> None:
    processed, models = tmp_path / "processed", tmp_path / "models"
    processed.mkdir()
    rows = []
    for match_index in range(8):
        split = "train" if match_index < 4 else "validation" if match_index < 6 else "test"
        for placement in range(1, 9):
            rows.append({
                "match_hash": f"m{match_index}", "participant_hash": f"p{match_index}-{placement}",
                "split": split, "top_four": placement <= 4, "placement": placement,
                "board_strength": 9 - placement, "composition_diversity": (placement + match_index) % 4,
            })
    fields = list(rows[0])
    write_csv_rows(processed / "features.csv", rows, fields)
    config = {
        "project": {"dataset_version": "test", "schema_version": "1"},
        "sampling": {"random_seed": 42},
        "features": {"prohibited_columns": ["match_hash", "participant_hash", "placement", "top_four"]},
        "model": {"primary_metric": "log_loss", "cross_validation_folds": 2,
                  "bootstrap_iterations": 10, "logistic_c_values": [1.0]},
        "paths": {"processed": str(processed), "models": str(models)},
    }
    run_dir = train_and_evaluate(config, experiment_id="test-experiment")
    report = load_json(run_dir / "metrics.json")
    assert report["selected_model"] == "logistic_c_1.0"
    assert 0 <= report["test_metrics"]["brier_score"] <= 1
    assert (run_dir / "fitted_pipeline.joblib").exists()
    assert (run_dir / "test_predictions.csv").exists()

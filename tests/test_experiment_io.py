import json

import pytest

from chicken_behavior_lab.io.experiment_io import (
    ExperimentOutputWriter,
)


def test_create_experiment_directory(tmp_path):
    writer = ExperimentOutputWriter(tmp_path)

    run_dir = writer.create_experiment_directory(
        experiment_name="baseline",
        timestamp="20261011T120000000000Z",
    )

    assert run_dir.is_dir()
    assert run_dir.name.startswith("baseline_")


def test_duplicate_run_directory_is_rejected(tmp_path):
    writer = ExperimentOutputWriter(tmp_path)

    writer.create_experiment_directory(
        experiment_name="baseline",
        timestamp="fixed",
    )

    with pytest.raises(FileExistsError):
        writer.create_experiment_directory(
            experiment_name="baseline",
            timestamp="fixed",
        )


def test_save_json_round_trip(tmp_path):
    path = ExperimentOutputWriter.save_json(
        tmp_path / "example.json",
        {"accuracy": 0.91, "labels": ["feeding", "resting"]},
    )

    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    assert payload["accuracy"] == pytest.approx(0.91)
    assert payload["labels"] == ["feeding", "resting"]


def test_prediction_errors_contains_only_mistakes(tmp_path):
    records = [
        {
            "sample_id": "s1",
            "true_index": 0,
            "predicted_index": 0,
        },
        {
            "sample_id": "s2",
            "true_index": 1,
            "predicted_index": 0,
        },
    ]

    path = ExperimentOutputWriter.save_prediction_errors(
        tmp_path,
        records,
    )

    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    assert payload["num_prediction_records"] == 2
    assert payload["num_errors"] == 1
    assert payload["errors"][0]["sample_id"] == "s2"


def test_nan_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        ExperimentOutputWriter.save_json(
            tmp_path / "invalid.json",
            {"metric": float("nan")},
        )

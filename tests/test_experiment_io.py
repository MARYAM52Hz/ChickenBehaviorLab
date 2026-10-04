from __future__ import annotations

import json

from chicken_behavior_lab.io.experiment_io import (
    ExperimentOutputWriter,
)


def test_experiment_directory_creation(
    tmp_path,
) -> None:

    writer = ExperimentOutputWriter(
        tmp_path
    )

    directory = (
        writer.create_experiment_directory(
            timestamp="20261004_120000"
        )
    )

    assert directory.exists()
    assert directory.is_dir()


def test_save_json(
    tmp_path,
) -> None:

    writer = ExperimentOutputWriter(
        tmp_path
    )

    directory = (
        writer.create_experiment_directory(
            timestamp="20261004_120001"
        )
    )

    path = writer.save_json(
        directory / "test.json",
        {
            "accuracy": 0.91,
            "labels": [
                "feeding",
                "walking",
            ],
        },
    )

    assert path.exists()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert data["accuracy"] == 0.91
    assert data["labels"] == [
        "feeding",
        "walking",
    ]

from pathlib import Path

from chicken_behavior_lab.io.experiment_io import (
    ExperimentOutputWriter,
)


def test_create_experiment_directory(
    tmp_path: Path,
) -> None:
    writer = ExperimentOutputWriter(
        root_directory=tmp_path
    )

    experiment_dir = (
        writer.create_experiment_directory(
            timestamp="20261005_211500"
        )
    )

    assert experiment_dir.exists()
    assert experiment_dir.is_dir()


def test_save_json(
    tmp_path: Path,
) -> None:
    writer = ExperimentOutputWriter(
        root_directory=tmp_path
    )

    experiment_dir = (
        writer.create_experiment_directory(
            timestamp="20261005_211501"
        )
    )

    output_path = writer.save_json(
        experiment_dir / "test.json",
        {
            "accuracy": 0.91,
            "classes": [
                "feeding",
                "walking",
            ],
        },
    )

    assert output_path.exists()

    content = output_path.read_text(
        encoding="utf-8"
    )

    assert "accuracy" in content
    assert "feeding" in content

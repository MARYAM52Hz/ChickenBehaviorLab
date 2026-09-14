import json

from chicken_behavior_lab.analysis import (
    TrainingHistoryAnalyzer,
)


def test_training_history_loading(
    tmp_path,
):

    path = (
        tmp_path
        / "training_history.json"
    )

    data = {
        "train_loss": [
            1.0,
            0.7,
            0.5,
        ],
        "validation_loss": [
            1.1,
            0.8,
            0.6,
        ],
        "validation_accuracy": [
            0.5,
            0.7,
            0.8,
        ],
        "validation_macro_f1": [
            0.45,
            0.68,
            0.79,
        ],
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
        )

    analyzer = (
        TrainingHistoryAnalyzer()
    )

    history = analyzer.load(
        path
    )

    assert len(
        history.epochs
    ) == 3

    assert (
        analyzer.best_epoch(history)
        == 3
    )

    assert (
        analyzer.best_validation_f1(
            history
        )
        == 0.79
    )

from __future__ import annotations

from dataclasses import dataclass

import torch

from chicken_behavior_lab.dataset.temporal_batch import (
    TemporalBatch,
)

from chicken_behavior_lab.training.temporal_experiment import (
    TemporalExperiment,
)

from chicken_behavior_lab.training.temporal_split import (
    GroupSplit,
)


class DummyTemporalDataset:
    def __init__(
        self,
        size: int,
    ) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self,
        index: int,
    ):
        raise NotImplementedError


class DummyTrainer:
    def __init__(self) -> None:
        self.label_mapping = {
            "feeding": 0,
            "walking": 1,
        }

    def fit(
        self,
        *,
        train_loader,
        validation_loader,
        epochs,
        save_every,
    ):
        return [
            {
                "epoch": 1.0,
                "train_loss": 0.5,
                "train_accuracy": 0.75,
                "val_loss": 0.6,
                "val_accuracy": 0.70,
            }
        ]


class DummyModel(torch.nn.Module):
    def forward(
        self,
        batch: TemporalBatch,
    ) -> torch.Tensor:

        batch_size = batch.batch_size

        logits = torch.zeros(
            batch_size,
            2,
            device=batch.x.device,
        )

        logits[:, 0] = 5.0

        return logits


class DummyTemporalEvaluator:
    pass


def test_temporal_experiment_initialization() -> None:
    split = GroupSplit(
        train_groups=("train",),
        validation_groups=("validation",),
        test_groups=("test",),
    )

    experiment = TemporalExperiment(
        model=DummyModel(),
        trainer=DummyTrainer(),
        train_dataset=DummyTemporalDataset(10),
        validation_dataset=DummyTemporalDataset(4),
        test_dataset=DummyTemporalDataset(4),
        split=split,
        batch_size=2,
    )

    assert experiment.batch_size == 2
    assert experiment.device.type == "cpu"

    assert experiment.train_dataset is not None
    assert experiment.validation_dataset is not None
    assert experiment.test_dataset is not None


def test_temporal_experiment_split_is_validated() -> None:
    split = GroupSplit(
        train_groups=("same",),
        validation_groups=("same",),
        test_groups=("test",),
    )

    try:
        TemporalExperiment(
            model=DummyModel(),
            trainer=DummyTrainer(),
            train_dataset=DummyTemporalDataset(1),
            validation_dataset=DummyTemporalDataset(1),
            test_dataset=DummyTemporalDataset(1),
            split=split,
        )
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError."
        )

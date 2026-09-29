from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from chicken_behavior_lab.dataset import (
    TemporalPyGDataset,
    TemporalCollator,
)

from chicken_behavior_lab.training.temporal_evaluator import (
    TemporalEvaluator,
)

from chicken_behavior_lab.training.temporal_split import (
    GroupSplit,
)


@dataclass(slots=True)
class TemporalExperimentResult:
    """
    Complete result of a temporal experiment.
    """

    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    test_metrics: dict[str, Any]

    history: list[dict[str, float]]

    split: GroupSplit

    train_size: int
    validation_size: int
    test_size: int


class TemporalExperiment:
    """
    Orchestrate training and evaluation of a temporal behavior model.

    The experiment assumes that datasets have already been split at the
    group level, before temporal windows are created.
    """

    def __init__(
        self,
        *,
        model,
        trainer,
        train_dataset: TemporalPyGDataset,
        validation_dataset: TemporalPyGDataset,
        test_dataset: TemporalPyGDataset,
        split: GroupSplit,
        batch_size: int = 8,
        device: str | torch.device = "cpu",
        num_workers: int = 0,
    ) -> None:

        if batch_size < 1:
            raise ValueError(
                "batch_size must be >= 1."
            )

        if num_workers < 0:
            raise ValueError(
                "num_workers must be >= 0."
            )

        self.model = model
        self.trainer = trainer

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        self.split = split
        self.split.validate()

        self.batch_size = batch_size
        self.device = torch.device(device)
        self.num_workers = num_workers

    def _make_loader(
        self,
        dataset,
        *,
        shuffle: bool,
    ):
        from torch.utils.data import DataLoader

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=TemporalCollator(),
        )

    def run(
        self,
        *,
        epochs: int,
        save_every: int = 1,
    ) -> TemporalExperimentResult:

        train_loader = self._make_loader(
            self.train_dataset,
            shuffle=True,
        )

        validation_loader = self._make_loader(
            self.validation_dataset,
            shuffle=False,
        )

        test_loader = self._make_loader(
            self.test_dataset,
            shuffle=False,
        )

        history = self.trainer.fit(
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=epochs,
            save_every=save_every,
        )

        train_metrics = (
            self.trainer.validate(
                train_loader
            )
        )

        validation_metrics = (
            self.trainer.validate(
                validation_loader
            )
        )

        index_to_label = {
            index: behavior_id
            for behavior_id, index
            in self.trainer.label_mapping.items()
        }

        evaluator = TemporalEvaluator(
            model=self.model,
            device=self.device,
            index_to_label=index_to_label,
        )

        test_result = evaluator.evaluate(
            test_loader
        )

        return TemporalExperimentResult(
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            test_metrics=test_result.metrics,
            history=history,
            split=self.split,
            train_size=len(
                self.train_dataset
            ),
            validation_size=len(
                self.validation_dataset
            ),
            test_size=len(
                self.test_dataset
            ),
        )

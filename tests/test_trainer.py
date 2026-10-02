from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from chicken_behavior_lab.training.trainer import (
    Trainer,
)


class TinyClassifier(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.model_type = "test"

        self.layer = nn.Linear(
            4,
            2,
        )

    def forward(
        self,
        batch,
    ):
        return self.layer(
            batch[0]
        )


def test_trainer_runs_one_epoch(
    tmp_path,
) -> None:

    x = torch.randn(
        12,
        4,
    )

    y = torch.randint(
        0,
        2,
        (12,),
    )

    dataset = TensorDataset(
        x,
        y,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
    )

    model = TinyClassifier()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device="cpu",
        checkpoint_dir=tmp_path,
        label_mapping={
            "class_a": 0,
            "class_b": 1,
        },
    )

    history = trainer.fit(
        train_loader=loader,
        epochs=1,
    )

    assert len(
        history["train_loss"]
    ) == 1

    assert len(
        history["train_accuracy"]
    ) == 1

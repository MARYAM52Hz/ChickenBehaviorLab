from __future__ import annotations

import torch
from torch import nn

from chicken_behavior_lab.training.checkpoint import (
    CheckpointManager,
    load_checkpoint,
)


class TinyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.model_type = "test"

        self.layer = nn.Linear(
            4,
            3,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.layer(x)


def test_checkpoint_save_and_load(
    tmp_path,
) -> None:

    model = TinyModel()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    manager = CheckpointManager(
        tmp_path,
        "best.pt",
    )

    path = manager.save(
        model=model,
        optimizer=optimizer,
        epoch=4,
        model_config={
            "model_type": "temporal",
            "node_feature_dim": 8,
            "num_classes": 3,
        },
        label_mapping={
            "feeding": 0,
            "walking": 1,
            "standing": 2,
        },
        best_metric=0.87,
        train_metrics={
            "loss": 0.42,
        },
        val_metrics={
            "loss": 0.31,
            "accuracy": 0.87,
        },
    )

    assert path.exists()

    checkpoint = manager.load(
        path
    )

    assert checkpoint["epoch"] == 4

    assert checkpoint["model_type"] == "test"

    assert checkpoint["label_mapping"] == {
        "feeding": 0,
        "walking": 1,
        "standing": 2,
    }

    assert checkpoint["best_metric"] == 0.87


def test_checkpoint_restore(
    tmp_path,
) -> None:

    model = TinyModel()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    manager = CheckpointManager(
        tmp_path
    )

    manager.save(
        model=model,
        optimizer=optimizer,
        epoch=3,
    )

    restored_model = TinyModel()

    restored_optimizer = torch.optim.Adam(
        restored_model.parameters(),
        lr=1e-3,
    )

    checkpoint = manager.load()

    epoch = manager.restore(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
    )

    assert epoch == 3


def test_load_checkpoint_function(
    tmp_path,
) -> None:

    model = TinyModel()

    manager = CheckpointManager(
        tmp_path,
        "model.pt",
    )

    manager.save(
        model=model,
        epoch=1,
    )

    checkpoint = load_checkpoint(
        tmp_path / "model.pt"
    )

    assert checkpoint["epoch"] == 1

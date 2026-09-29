from __future__ import annotations

import torch
import torch.nn as nn

from chicken_behavior_lab.config.model_config import (
    ModelConfig,
)

from chicken_behavior_lab.models.model_factory import (
    build_model,
)

from chicken_behavior_lab.training.trainer import (
    Trainer,
)


def build_trainer(
    *,
    model_config: ModelConfig,
    label_mapping: dict[str, int],
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | torch.device = "cpu",
    checkpoint_dir: str = "checkpoints",
    scheduler=None,
) -> Trainer:
    """
    Build a model, optimizer, loss function, and Trainer.

    Parameters
    ----------
    model_config:
        Complete model configuration.

    label_mapping:
        Mapping from behavior_id to integer class index.

    learning_rate:
        AdamW learning rate.

    weight_decay:
        AdamW weight decay.

    device:
        Training device.

    checkpoint_dir:
        Directory for checkpoints.

    scheduler:
        Optional learning-rate scheduler.
    """

    if not label_mapping:
        raise ValueError(
            "label_mapping cannot be empty."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be > 0."
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay must be >= 0."
        )

    model = build_model(
        model_config
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        model_config=model_config,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=checkpoint_dir,
        label_mapping=label_mapping,
        scheduler=scheduler,
    )

    return trainer

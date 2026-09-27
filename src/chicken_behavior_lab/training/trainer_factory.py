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
) -> Trainer:

    model = build_model(
        model_config
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    return Trainer(
        model=model,
        model_config=model_config,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=checkpoint_dir,
        label_mapping=label_mapping,
    )

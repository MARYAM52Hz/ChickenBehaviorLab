from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chicken_behavior_lab.training.trainer import Trainer


def build_trainer(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    model_config: Any | None = None,
    loss_function: nn.Module | None = None,
    criterion: nn.Module | None = None,
    label_mapping: dict[str, int] | None = None,
    label_to_index: dict[str, int] | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_path: str | None = None,
    training_config: dict[str, Any] | None = None,
    scheduler: Any | None = None,
) -> Trainer:

    return Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        model_config=model_config,
        loss_function=loss_function,
        criterion=criterion,
        label_mapping=label_mapping,
        label_to_index=label_to_index,
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        training_config=training_config,
        scheduler=scheduler,
    )

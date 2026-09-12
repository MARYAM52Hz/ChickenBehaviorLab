from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    epoch: int | None = None,
    train_loss: float | None = None,
    validation_loss: float | None = None,
    validation_f1: float | None = None,
    model_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    label_to_index: dict[str, int] | None = None,
) -> None:
    """
    Save a complete training checkpoint.

    The checkpoint contains model weights together with
    the information required to reproduce the experiment.
    """

    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint[
            "optimizer_state_dict"
        ] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if train_loss is not None:
        checkpoint["train_loss"] = train_loss

    if validation_loss is not None:
        checkpoint[
            "validation_loss"
        ] = validation_loss

    if validation_f1 is not None:
        checkpoint[
            "validation_f1"
        ] = validation_f1

    if model_config is not None:
        checkpoint[
            "model_config"
        ] = model_config

    if training_config is not None:
        checkpoint[
            "training_config"
        ] = training_config

    if label_to_index is not None:
        checkpoint[
            "label_to_index"
        ] = label_to_index

    checkpoint_path = Path(path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        checkpoint_path,
    )


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Load a complete training checkpoint.
    """

    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ValueError(
            "Checkpoint must contain "
            "a dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    return checkpoint

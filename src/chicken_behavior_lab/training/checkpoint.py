from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from chicken_behavior_lab.config.model_config import (
    ModelConfig,
)


CHECKPOINT_FORMAT_VERSION = 1


class CheckpointManager:
    """
    Save and load complete training checkpoints.

    A checkpoint contains everything required to reconstruct
    the model and continue training or perform evaluation.
    """

    def __init__(
        self,
        directory: str | Path,
    ) -> None:

        self.directory = Path(directory)

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        *,
        filename: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        epoch: int,
        model_config: ModelConfig,
        label_mapping: dict[str, int],
        best_metric: float | None = None,
        train_metrics: dict[str, float] | None = None,
        val_metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:

        model_config.validate()

        checkpoint = {
            "format_version": CHECKPOINT_FORMAT_VERSION,

            "epoch": int(epoch),

            "model_type": model_config.model_type,

            "model_config": {
                key: value
                for key, value in vars(
                    model_config
                ).items()
            },

            "label_mapping": dict(
                label_mapping
            ),

            "model_state_dict": (
                model.state_dict()
            ),

            "optimizer_state_dict": (
                optimizer.state_dict()
                if optimizer is not None
                else None
            ),

            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),

            "best_metric": (
                float(best_metric)
                if best_metric is not None
                else None
            ),

            "train_metrics": (
                dict(train_metrics)
                if train_metrics is not None
                else {}
            ),

            "val_metrics": (
                dict(val_metrics)
                if val_metrics is not None
                else {}
            ),

            "metadata": (
                dict(metadata)
                if metadata is not None
                else {}
            ),
        }

        path = self.directory / filename

        torch.save(
            checkpoint,
            path,
        )

        return path

    def load(
        self,
        filename: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:

        path = Path(filename)

        if not path.is_absolute():
            path = self.directory / path

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

        self._validate_checkpoint(
            checkpoint
        )

        return checkpoint

    @staticmethod
    def _validate_checkpoint(
        checkpoint: Any,
    ) -> None:

        if not isinstance(
            checkpoint,
            dict,
        ):
            raise ValueError(
                "Checkpoint must contain a dictionary."
            )

        required_keys = {
            "format_version",
            "epoch",
            "model_type",
            "model_config",
            "label_mapping",
            "model_state_dict",
        }

        missing = (
            required_keys
            - checkpoint.keys()
        )

        if missing:
            raise ValueError(
                "Checkpoint is missing required "
                f"keys: {sorted(missing)}"
            )

        if (
            checkpoint["format_version"]
            > CHECKPOINT_FORMAT_VERSION
        ):
            raise ValueError(
                "Checkpoint format is newer than "
                "the current code."
            )

    @staticmethod
    def build_model_config(
        checkpoint: dict[str, Any],
    ) -> ModelConfig:

        config_dict = dict(
            checkpoint["model_config"]
        )

        config = ModelConfig(
            **config_dict
        )

        config.validate()

        return config

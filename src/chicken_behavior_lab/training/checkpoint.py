from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from chicken_behavior_lab.config.model_config import ModelConfig


CHECKPOINT_FORMAT_VERSION = 1


class CheckpointManager:
    """
    Save and load model training checkpoints.

    A checkpoint stores:
        - checkpoint format version
        - epoch
        - model type
        - model configuration
        - label mapping
        - model state
        - optimizer state
        - scheduler state
        - best validation metric
        - training metrics
        - validation metrics
        - additional metadata
    """

    REQUIRED_KEYS = {
        "format_version",
        "epoch",
        "model_type",
        "model_config",
        "label_mapping",
        "model_state_dict",
        "best_metric",
        "train_metrics",
        "val_metrics",
        "metadata",
    }

    def __init__(
        self,
        directory: str | Path = "checkpoints",
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
        label_mapping: dict[str, int] | None = None,
        best_metric: float | None = None,
        train_metrics: dict[str, float] | None = None,
        val_metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        if epoch < 0:
            raise ValueError("epoch must be >= 0.")

        if not filename:
            raise ValueError("filename cannot be empty.")

        if not isinstance(model_config, ModelConfig):
            raise TypeError(
                "model_config must be an instance of ModelConfig."
            )

        checkpoint = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "epoch": int(epoch),
            "model_type": model_config.model_type,
            "model_config": asdict(model_config),
            "label_mapping": dict(label_mapping or {}),
            "model_state_dict": model.state_dict(),
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
            "train_metrics": dict(train_metrics or {}),
            "val_metrics": dict(val_metrics or {}),
            "metadata": dict(metadata or {}),
        }

        path = self.directory / filename
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            checkpoint,
            path,
        )

        return path

    def load(
        self,
        filename: str | Path,
        *,
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
        )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "Checkpoint must contain a dictionary."
            )

        self._validate_checkpoint(checkpoint)

        return checkpoint

    @classmethod
    def _validate_checkpoint(
        cls,
        checkpoint: dict[str, Any],
    ) -> None:
        missing_keys = cls.REQUIRED_KEYS.difference(
            checkpoint.keys()
        )

        if missing_keys:
            raise ValueError(
                "Checkpoint is missing required keys: "
                + ", ".join(sorted(missing_keys))
            )

        if checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "Unsupported checkpoint format version: "
                f"{checkpoint['format_version']}"
            )

        if not isinstance(
            checkpoint["model_config"],
            dict,
        ):
            raise TypeError(
                "checkpoint['model_config'] must be a dictionary."
            )

        if not isinstance(
            checkpoint["label_mapping"],
            dict,
        ):
            raise TypeError(
                "checkpoint['label_mapping'] must be a dictionary."
            )

        if not isinstance(
            checkpoint["model_state_dict"],
            dict,
        ):
            raise TypeError(
                "checkpoint['model_state_dict'] must be a dictionary."
            )

    @staticmethod
    def build_model_config(
        checkpoint: dict[str, Any],
    ) -> ModelConfig:
        if "model_config" not in checkpoint:
            raise ValueError(
                "Checkpoint does not contain model_config."
            )

        config_dict = dict(
            checkpoint["model_config"]
        )

        return ModelConfig(
            **config_dict
        )


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Backward-compatible convenience function.
    """

    path = Path(path)

    manager = CheckpointManager(
        path.parent
    )

    return manager.load(
        path.name,
        map_location=map_location,
    )

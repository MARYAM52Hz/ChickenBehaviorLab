from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FORMAT_VERSION = 1


class CheckpointManager:
    """
    Save and load complete training checkpoints.

    The checkpoint contains:
        - model state
        - optimizer state
        - scheduler state
        - epoch
        - model configuration
        - label mapping
        - training/validation metrics
        - arbitrary metadata
    """

    def __init__(
        self,
        directory: str | Path,
        filename: str = "checkpoint.pt",
    ) -> None:
        self.directory = Path(directory)
        self.filename = filename

        if not self.filename.endswith(".pt"):
            raise ValueError("Checkpoint filename must end with '.pt'.")

        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self.directory / self.filename

    def save(
        self,
        *,
        model: torch.nn.Module,
        epoch: int,
        model_config: Any | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        label_mapping: dict[str, int] | None = None,
        best_metric: float | None = None,
        train_metrics: dict[str, Any] | None = None,
        val_metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        path: str | Path | None = None,
    ) -> Path:
        if epoch < 0:
            raise ValueError("epoch must be >= 0.")

        checkpoint_path = (
            Path(path)
            if path is not None
            else self.path
        )

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "epoch": int(epoch),
            "model_type": self._infer_model_type(model),
            "model_config": self._serialize_config(model_config),
            "label_mapping": (
                dict(label_mapping)
                if label_mapping is not None
                else None
            ),
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

        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path

    def load(
        self,
        path: str | Path | None = None,
        *,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        checkpoint_path = (
            Path(path)
            if path is not None
            else self.path
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
        )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "Checkpoint must contain a dictionary."
            )

        self._validate_checkpoint(checkpoint)

        return checkpoint

    def restore(
        self,
        checkpoint: dict[str, Any],
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        strict: bool = True,
    ) -> int:
        self._validate_checkpoint(checkpoint)

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=strict,
        )

        optimizer_state = checkpoint.get(
            "optimizer_state_dict"
        )

        if (
            optimizer is not None
            and optimizer_state is not None
        ):
            optimizer.load_state_dict(
                optimizer_state
            )

        scheduler_state = checkpoint.get(
            "scheduler_state_dict"
        )

        if (
            scheduler is not None
            and scheduler_state is not None
        ):
            scheduler.load_state_dict(
                scheduler_state
            )

        return int(checkpoint["epoch"])

    @staticmethod
    def _serialize_config(config: Any | None) -> Any:
        if config is None:
            return None

        if is_dataclass(config):
            return asdict(config)

        if isinstance(config, dict):
            return dict(config)

        if hasattr(config, "to_dict"):
            return dict(config.to_dict())

        raise TypeError(
            "model_config must be a dataclass, dictionary, "
            "or provide a to_dict() method."
        )

    @staticmethod
    def _infer_model_type(
        model: torch.nn.Module,
    ) -> str | None:
        if hasattr(model, "model_type"):
            return str(model.model_type)

        class_name = model.__class__.__name__

        if class_name == "TemporalBehaviorGNN":
            return "temporal"

        if class_name == "ChickenBehaviorGNN":
            return "baseline"

        return None

    @staticmethod
    def _validate_checkpoint(
        checkpoint: dict[str, Any],
    ) -> None:
        required_keys = {
            "format_version",
            "epoch",
            "model_state_dict",
        }

        missing = required_keys.difference(
            checkpoint.keys()
        )

        if missing:
            raise ValueError(
                "Checkpoint is missing required fields: "
                f"{sorted(missing)}"
            )

        version = checkpoint["format_version"]

        if version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "Unsupported checkpoint format version: "
                f"{version}. Expected "
                f"{CHECKPOINT_FORMAT_VERSION}."
            )

        if not isinstance(
            checkpoint["epoch"],
            int,
        ):
            raise TypeError(
                "Checkpoint epoch must be an integer."
            )


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Backward-compatible convenience function.
    """
    manager = CheckpointManager(
        directory=Path(path).parent,
        filename=Path(path).name,
    )

    return manager.load(
        map_location=map_location,
    )

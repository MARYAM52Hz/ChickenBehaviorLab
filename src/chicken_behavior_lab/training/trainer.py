from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chicken_behavior_lab.dataset.temporal_batch import TemporalBatch
from chicken_behavior_lab.training.checkpoint import CheckpointManager


class Trainer:
    """
    Generic trainer supporting both graph and temporal models.

    Backward-compatible constructor aliases:
        criterion <-> loss_function
        label_mapping <-> label_to_index
        checkpoint_dir <-> checkpoint_path
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str,

        criterion: nn.Module | None = None,
        loss_function: nn.Module | None = None,

        checkpoint_dir: str | None = None,
        checkpoint_path: str | None = None,

        model_config: Any | None = None,

        label_mapping: dict[str, int] | None = None,
        label_to_index: dict[str, int] | None = None,

        training_config: dict[str, Any] | None = None,

        scheduler: Any | None = None,

        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.model.to(self.device)

        if criterion is not None and loss_function is not None:
            if criterion is not loss_function:
                raise ValueError(
                    "Provide only one of criterion or loss_function."
                )

        self.criterion = (
            criterion
            if criterion is not None
            else loss_function
        )

        if self.criterion is None:
            raise ValueError(
                "A loss function must be provided through "
                "'criterion' or 'loss_function'."
            )

        if (
            label_mapping is not None
            and label_to_index is not None
            and label_mapping != label_to_index
        ):
            raise ValueError(
                "label_mapping and label_to_index disagree."
            )

        self.label_mapping = dict(
            label_mapping
            if label_mapping is not None
            else label_to_index or {}
        )

        self.model_config = model_config
        self.training_config = dict(
            training_config or {}
        )

        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = None

        if (
            checkpoint_manager is not None
            and (
                checkpoint_dir is not None
                or checkpoint_path is not None
            )
        ):
            raise ValueError(
                "Provide checkpoint_manager or checkpoint path, "
                "not both."
            )

        if checkpoint_manager is not None:
            self.checkpoint_manager = checkpoint_manager

        elif checkpoint_path is not None:
            path = checkpoint_path

            from pathlib import Path

            path_obj = Path(path)

            self.checkpoint_manager = CheckpointManager(
                directory=path_obj.parent,
                filename=path_obj.name,
            )

        elif checkpoint_dir is not None:
            self.checkpoint_manager = CheckpointManager(
                directory=checkpoint_dir,
            )

        else:
            self.checkpoint_manager = None

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        self.best_metric: float | None = None
        self.best_epoch: int | None = None

    def fit(
        self,
        train_loader,
        validation_loader=None,
        epochs: int = 1,
        val_loader=None,
    ) -> dict[str, list[float]]:
        """
        Train the model.

        val_loader is accepted as an alias for validation_loader.
        """

        if validation_loader is not None and val_loader is not None:
            raise ValueError(
                "Provide only validation_loader or val_loader."
            )

        if validation_loader is None:
            validation_loader = val_loader

        if epochs < 1:
            raise ValueError(
                "epochs must be >= 1."
            )

        for epoch in range(1, epochs + 1):

            train_metrics = self._train_epoch(
                train_loader
            )

            if validation_loader is not None:
                val_metrics = self._evaluate_loader(
                    validation_loader
                )
            else:
                val_metrics = {
                    "loss": float("nan"),
                    "accuracy": float("nan"),
                }

            self.history["train_loss"].append(
                train_metrics["loss"]
            )

            self.history["train_accuracy"].append(
                train_metrics["accuracy"]
            )

            self.history["val_loss"].append(
                val_metrics["loss"]
            )

            self.history["val_accuracy"].append(
                val_metrics["accuracy"]
            )

            if self.scheduler is not None:
                self.scheduler.step()

            self._update_best_checkpoint(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

        return self.history

    def _train_epoch(
        self,
        data_loader,
    ) -> dict[str, float]:

        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in data_loader:

            batch = self._move_batch_to_device(
                batch
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            logits = self._forward(batch)

            target = self._extract_target(
                batch
            )

            loss = self.criterion(
                logits,
                target,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Training loss became non-finite."
                )

            loss.backward()

            self.optimizer.step()

            batch_size = target.shape[0]

            total_loss += (
                float(loss.detach().item())
                * batch_size
            )

            predictions = torch.argmax(
                logits,
                dim=-1,
            )

            total_correct += int(
                (predictions == target)
                .sum()
                .item()
            )

            total_samples += batch_size

        if total_samples == 0:
            raise ValueError(
                "Training loader produced no samples."
            )

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    @torch.no_grad()
    def _evaluate_loader(
        self,
        data_loader,
    ) -> dict[str, float]:

        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in data_loader:

            batch = self._move_batch_to_device(
                batch
            )

            logits = self._forward(batch)

            target = self._extract_target(
                batch
            )

            loss = self.criterion(
                logits,
                target,
            )

            batch_size = target.shape[0]

            total_loss += (
                float(loss.detach().item())
                * batch_size
            )

            predictions = torch.argmax(
                logits,
                dim=-1,
            )

            total_correct += int(
                (predictions == target)
                .sum()
                .item()
            )

            total_samples += batch_size

        if total_samples == 0:
            raise ValueError(
                "Validation loader produced no samples."
            )

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    def _forward(
        self,
        batch,
    ) -> torch.Tensor:

        if isinstance(batch, TemporalBatch):
            return self.model(batch)

        return self.model(batch)

    @staticmethod
    def _extract_target(
        batch,
    ) -> torch.Tensor:

        if isinstance(batch, TemporalBatch):
            return batch.y.long()

        if hasattr(batch, "y"):
            target = batch.y

            if target.ndim > 1:
                target = target.view(-1)

            return target.long()

        raise TypeError(
            "Batch must provide a 'y' target tensor."
        )

    def _move_batch_to_device(
        self,
        batch,
    ):
        if hasattr(batch, "to"):
            return batch.to(self.device)

        raise TypeError(
            "Batch object must implement .to(device)."
        )

    def _update_best_checkpoint(
        self,
        *,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:

        metric = val_metrics["accuracy"]

        if metric != metric:
            metric = train_metrics["accuracy"]

        is_better = (
            self.best_metric is None
            or metric > self.best_metric
        )

        if not is_better:
            return

        self.best_metric = float(metric)
        self.best_epoch = epoch

        if self.checkpoint_manager is None:
            return

        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            model_config=self.model_config,
            label_mapping=self.label_mapping,
            best_metric=self.best_metric,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            metadata={
                "training_config": self.training_config,
                "best_epoch": self.best_epoch,
            },
        )

    def save_checkpoint(
        self,
        epoch: int,
        train_metrics: dict[str, Any] | None = None,
        val_metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if self.checkpoint_manager is None:
            raise RuntimeError(
                "No checkpoint manager configured."
            )

        return self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            model_config=self.model_config,
            label_mapping=self.label_mapping,
            best_metric=self.best_metric,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            metadata={
                "training_config": self.training_config,
                **(metadata or {}),
            },
        )

    def load_checkpoint(
        self,
        path: str | None = None,
        *,
        strict: bool = True,
    ) -> int:

        if self.checkpoint_manager is None:
            raise RuntimeError(
                "No checkpoint manager configured."
            )

        checkpoint = self.checkpoint_manager.load(
            path=path,
            map_location=self.device,
        )

        epoch = self.checkpoint_manager.restore(
            checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            strict=strict,
        )

        best_metric = checkpoint.get(
            "best_metric"
        )

        if best_metric is not None:
            self.best_metric = float(
                best_metric
            )

        metadata = checkpoint.get(
            "metadata",
            {},
        )

        if isinstance(metadata, dict):
            best_epoch = metadata.get(
                "best_epoch"
            )

            if best_epoch is not None:
                self.best_epoch = int(
                    best_epoch
                )

        return epoch

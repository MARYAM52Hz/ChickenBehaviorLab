from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from chicken_behavior_lab.config.model_config import (
    ModelConfig,
)

from chicken_behavior_lab.dataset.temporal_batch import (
    TemporalBatch,
)

from chicken_behavior_lab.training.checkpoint import (
    CheckpointManager,
)


class Trainer:
    """
    Training loop for temporal behavior models.

    Responsibilities
    ----------------
    - forward pass
    - loss computation
    - backward pass
    - optimizer step
    - validation
    - checkpointing
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        model_config: ModelConfig,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device | str = "cpu",
        checkpoint_dir: str | Path = "checkpoints",
        label_mapping: dict[str, int] | None = None,
        scheduler=None,
        metric_fn: Callable[
            [torch.Tensor, torch.Tensor],
            float,
        ] | None = None,
    ) -> None:

        self.model = model

        self.model_config = model_config

        self.optimizer = optimizer

        self.criterion = criterion

        self.device = torch.device(
            device
        )

        self.scheduler = scheduler

        self.label_mapping = (
            label_mapping or {}
        )

        self.metric_fn = metric_fn

        self.checkpoints = (
            CheckpointManager(
                checkpoint_dir
            )
        )

        self.model.to(
            self.device
        )

        self.current_epoch = 0

        self.best_metric = None

    def train_epoch(
        self,
        loader,
    ) -> dict[str, float]:

        self.model.train()

        total_loss = 0.0

        total_samples = 0

        predictions = []

        targets = []

        for batch in loader:

            batch = batch.to(
                self.device
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            logits = self.model(
                batch
            )

            loss = self.criterion(
                logits,
                batch.y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )

            self.optimizer.step()

            batch_size = (
                batch.y.shape[0]
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

            predictions.append(
                logits.detach()
            )

            targets.append(
                batch.y.detach()
            )

        if total_samples == 0:
            raise RuntimeError(
                "Training loader produced no samples."
            )

        metrics = {
            "loss": (
                total_loss
                / total_samples
            )
        }

        if predictions:
            predictions_tensor = torch.cat(
                predictions,
                dim=0,
            )

            targets_tensor = torch.cat(
                targets,
                dim=0,
            )

            metrics["accuracy"] = (
                self._accuracy(
                    predictions_tensor,
                    targets_tensor,
                )
            )

            if self.metric_fn is not None:
                metrics["custom_metric"] = (
                    float(
                        self.metric_fn(
                            predictions_tensor,
                            targets_tensor,
                        )
                    )
                )

        return metrics

    @torch.no_grad()
    def validate(
        self,
        loader,
    ) -> dict[str, float]:

        self.model.eval()

        total_loss = 0.0

        total_samples = 0

        predictions = []

        targets = []

        for batch in loader:

            batch = batch.to(
                self.device
            )

            logits = self.model(
                batch
            )

            loss = self.criterion(
                logits,
                batch.y,
            )

            batch_size = (
                batch.y.shape[0]
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

            predictions.append(
                logits
            )

            targets.append(
                batch.y
            )

        if total_samples == 0:
            raise RuntimeError(
                "Validation loader produced no samples."
            )

        predictions_tensor = torch.cat(
            predictions,
            dim=0,
        )

        targets_tensor = torch.cat(
            targets,
            dim=0,
        )

        metrics = {
            "loss": (
                total_loss
                / total_samples
            ),
            "accuracy": (
                self._accuracy(
                    predictions_tensor,
                    targets_tensor,
                )
            ),
        }

        if self.metric_fn is not None:
            metrics["custom_metric"] = (
                float(
                    self.metric_fn(
                        predictions_tensor,
                        targets_tensor,
                    )
                )
            )

        return metrics

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 10,
        save_every: int = 1,
    ) -> list[dict[str, float]]:

        if epochs < 1:
            raise ValueError(
                "epochs must be >= 1."
            )

        if save_every < 1:
            raise ValueError(
                "save_every must be >= 1."
            )

        history = []

        for epoch in range(
            self.current_epoch + 1,
            epochs + 1,
        ):

            self.current_epoch = epoch

            train_metrics = (
                self.train_epoch(
                    train_loader
                )
            )

            if val_loader is not None:
                val_metrics = (
                    self.validate(
                        val_loader
                    )
                )
            else:
                val_metrics = {}

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_record = {
                "epoch": float(epoch),
                "train_loss": train_metrics[
                    "loss"
                ],
                "train_accuracy": train_metrics[
                    "accuracy"
                ],
            }

            if val_metrics:
                epoch_record[
                    "val_loss"
                ] = val_metrics["loss"]

                epoch_record[
                    "val_accuracy"
                ] = val_metrics["accuracy"]

            history.append(
                epoch_record
            )

            if (
                val_metrics
                and "accuracy" in val_metrics
            ):

                current_metric = (
                    val_metrics["accuracy"]
                )

                is_best = (
                    self.best_metric is None
                    or current_metric
                    > self.best_metric
                )

                if is_best:

                    self.best_metric = (
                        current_metric
                    )

                    self.save_checkpoint(
                        filename="best.pt",
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                    )

            if (
                epoch % save_every
                == 0
            ):

                self.save_checkpoint(
                    filename=(
                        f"epoch_{epoch:04d}.pt"
                    ),
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                )

        return history

    def save_checkpoint(
        self,
        *,
        filename: str,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> Path:

        return self.checkpoints.save(
            filename=filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            model_config=self.model_config,
            label_mapping=self.label_mapping,
            best_metric=self.best_metric,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            metadata={
                "device": str(
                    self.device
                ),
            },
        )

    @staticmethod
    def _accuracy(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:

        predictions = (
            logits.argmax(
                dim=-1
            )
        )

        correct = (
            predictions
            == targets
        ).sum().item()

        total = targets.numel()

        if total == 0:
            return 0.0

        return (
            correct / total
        )

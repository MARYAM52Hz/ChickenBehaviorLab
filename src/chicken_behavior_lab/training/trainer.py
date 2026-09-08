from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader

from chicken_behavior_lab.training.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
)


@dataclass(frozen=True, slots=True)
class EpochResult:
    """
    Metrics collected for one epoch.
    """

    loss: float

    accuracy: float

    macro_precision: float

    macro_recall: float

    macro_f1: float


@dataclass(slots=True)
class TrainingHistory:
    """
    Complete training history.
    """

    train: list[EpochResult]

    validation: list[EpochResult]

    best_epoch: int | None = None

    best_validation_f1: float | None = None


class Trainer:
    """
    Training engine for ChickenBehaviorLab models.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        num_classes: int,
        device: str | torch.device,
        checkpoint_path: str | Path | None = None,
    ) -> None:

        if num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than 1."
            )

        self.model = model

        self.optimizer = optimizer

        self.loss_fn = loss_fn

        self.num_classes = (
            num_classes
        )

        self.device = torch.device(
            device
        )

        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else None
        )

        self.model.to(
            self.device
        )

    # =====================================================
    # Train epoch
    # =====================================================

    def train_epoch(
        self,
        loader: DataLoader,
    ) -> EpochResult:
        """
        Run one training epoch.
        """

        self.model.train()

        total_loss = 0.0

        total_samples = 0

        all_predictions: list[Tensor] = []

        all_targets: list[Tensor] = []

        for batch in loader:

            batch = batch.to(
                self.device
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            logits = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
            )

            targets = batch.y.view(
                -1
            )

            loss = self.loss_fn(
                logits,
                targets,
            )

            loss.backward()

            self.optimizer.step()

            batch_size = (
                targets.shape[0]
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

            predictions = (
                logits.argmax(
                    dim=1
                )
            )

            all_predictions.append(
                predictions.detach().cpu()
            )

            all_targets.append(
                targets.detach().cpu()
            )

        return self._build_epoch_result(
            total_loss=total_loss,
            total_samples=total_samples,
            predictions=all_predictions,
            targets=all_targets,
        )

    # =====================================================
    # Validation
    # =====================================================

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
    ) -> EpochResult:
        """
        Evaluate model without gradient computation.
        """

        self.model.eval()

        total_loss = 0.0

        total_samples = 0

        all_predictions: list[Tensor] = []

        all_targets: list[Tensor] = []

        for batch in loader:

            batch = batch.to(
                self.device
            )

            logits = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
            )

            targets = batch.y.view(
                -1
            )

            loss = self.loss_fn(
                logits,
                targets,
            )

            batch_size = (
                targets.shape[0]
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

            predictions = (
                logits.argmax(
                    dim=1
                )
            )

            all_predictions.append(
                predictions.cpu()
            )

            all_targets.append(
                targets.cpu()
            )

        return self._build_epoch_result(
            total_loss=total_loss,
            total_samples=total_samples,
            predictions=all_predictions,
            targets=all_targets,
        )

    # =====================================================
    # Fit
    # =====================================================

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int,
    ) -> TrainingHistory:
        """
        Train model for multiple epochs.

        Best checkpoint is selected using validation
        macro-F1 rather than validation accuracy.
        """

        if epochs <= 0:
            raise ValueError(
                "epochs must be positive."
            )

        history = TrainingHistory(
            train=[],
            validation=[],
        )

        best_f1 = float(
            "-inf"
        )

        for epoch in range(
            1,
            epochs + 1,
        ):

            train_result = (
                self.train_epoch(
                    train_loader
                )
            )

            validation_result = (
                self.evaluate(
                    validation_loader
                )
            )

            history.train.append(
                train_result
            )

            history.validation.append(
                validation_result
            )

            print(
                self._format_epoch(
                    epoch=epoch,
                    epochs=epochs,
                    train_result=train_result,
                    validation_result=(
                        validation_result
                    ),
                )
            )

            if (
                validation_result.macro_f1
                > best_f1
            ):

                best_f1 = (
                    validation_result.macro_f1
                )

                history.best_epoch = (
                    epoch
                )

                history.best_validation_f1 = (
                    best_f1
                )

                self.save_checkpoint(
                    epoch=epoch,
                    validation_result=(
                        validation_result
                    ),
                )

        return history

    # =====================================================
    # Metrics helper
    # =====================================================

    def _build_epoch_result(
        self,
        total_loss: float,
        total_samples: int,
        predictions: list[Tensor],
        targets: list[Tensor],
    ) -> EpochResult:

        if total_samples == 0:
            raise ValueError(
                "DataLoader contains no samples."
            )

        prediction_tensor = torch.cat(
            predictions
        )

        target_tensor = torch.cat(
            targets
        )

        metrics: ClassificationMetrics = (
            compute_classification_metrics(
                predictions=prediction_tensor,
                targets=target_tensor,
                num_classes=self.num_classes,
            )
        )

        average_loss = (
            total_loss
            / total_samples
        )

        return EpochResult(
            loss=float(
                average_loss
            ),
            accuracy=metrics.accuracy,
            macro_precision=(
                metrics.macro_precision
            ),
            macro_recall=(
                metrics.macro_recall
            ),
            macro_f1=(
                metrics.macro_f1
            ),
        )

    # =====================================================
    # Checkpoint
    # =====================================================

    def save_checkpoint(
        self,
        epoch: int,
        validation_result: EpochResult,
    ) -> None:
        """
        Save the current model if checkpointing
        is enabled.
        """

        if self.checkpoint_path is None:
            return

        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": (
                self.model.state_dict()
            ),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
            "validation_loss": (
                validation_result.loss
            ),
            "validation_accuracy": (
                validation_result.accuracy
            ),
            "validation_macro_f1": (
                validation_result.macro_f1
            ),
            "num_classes": (
                self.num_classes
            ),
        }

        torch.save(
            checkpoint,
            self.checkpoint_path,
        )

    # =====================================================
    # Load checkpoint
    # =====================================================

    def load_checkpoint(
        self,
        path: str | Path | None = None,
    ) -> dict:

        checkpoint_path = (
            Path(path)
            if path is not None
            else self.checkpoint_path
        )

        if checkpoint_path is None:
            raise ValueError(
                "No checkpoint path provided."
            )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        return checkpoint

    # =====================================================
    # Logging
    # =====================================================

    @staticmethod
    def _format_epoch(
        epoch: int,
        epochs: int,
        train_result: EpochResult,
        validation_result: EpochResult,
    ) -> str:

        return (
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train Loss: "
            f"{train_result.loss:.4f} | "
            f"Train Acc: "
            f"{train_result.accuracy:.4f} | "
            f"Train F1: "
            f"{train_result.macro_f1:.4f} | "
            f"Val Loss: "
            f"{validation_result.loss:.4f} | "
            f"Val Acc: "
            f"{validation_result.accuracy:.4f} | "
            f"Val F1: "
            f"{validation_result.macro_f1:.4f}"
        )

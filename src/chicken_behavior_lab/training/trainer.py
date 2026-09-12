from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader

from chicken_behavior_lab.training.checkpoint import (
    save_checkpoint,
)


@dataclass(slots=True)
class TrainingHistory:

    train_loss: list[float] = field(
        default_factory=list
    )

    validation_loss: list[float] = field(
        default_factory=list
    )

    validation_accuracy: list[float] = field(
        default_factory=list
    )

    validation_macro_f1: list[float] = field(
        default_factory=list
    )


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_function: nn.Module,
        device: torch.device,
        checkpoint_path: str,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        label_to_index: dict[str, int],
    ) -> None:

        self.model = model

        self.optimizer = optimizer

        self.loss_function = loss_function

        self.device = device

        self.checkpoint_path = (
            checkpoint_path
        )

        self.model_config = (
            model_config
        )

        self.training_config = (
            training_config
        )

        self.label_to_index = (
            label_to_index
        )

        self.history = (
            TrainingHistory()
        )

        self.best_validation_f1 = float(
            "-inf"
        )

    def train_epoch(
        self,
        data_loader: DataLoader,
    ) -> float:

        self.model.train()

        total_loss = 0.0

        total_samples = 0

        for batch in data_loader:

            batch = batch.to(
                self.device
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            logits = self.model(
                batch
            )

            targets = batch.y.view(
                -1
            )

            loss = self.loss_function(
                logits,
                targets,
            )

            loss.backward()

            self.optimizer.step()

            batch_size = (
                targets.size(0)
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

        if total_samples == 0:
            raise RuntimeError(
                "Training DataLoader "
                "returned no samples."
            )

        return (
            total_loss
            / total_samples
        )

    @torch.no_grad()
    def validate(
        self,
        data_loader: DataLoader,
    ) -> tuple[
        float,
        float,
        float,
    ]:

        self.model.eval()

        total_loss = 0.0

        total_samples = 0

        all_targets = []

        all_predictions = []

        for batch in data_loader:

            batch = batch.to(
                self.device
            )

            logits = self.model(
                batch
            )

            targets = batch.y.view(
                -1
            )

            loss = self.loss_function(
                logits,
                targets,
            )

            predictions = (
                torch.argmax(
                    logits,
                    dim=-1,
                )
            )

            batch_size = (
                targets.size(0)
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

            all_targets.append(
                targets.detach().cpu()
            )

            all_predictions.append(
                predictions.detach().cpu()
            )

        if total_samples == 0:
            raise RuntimeError(
                "Validation DataLoader "
                "returned no samples."
            )

        y_true = torch.cat(
            all_targets
        )

        y_pred = torch.cat(
            all_predictions
        )

        accuracy = float(
            (
                y_true == y_pred
            ).float().mean().item()
        )

        num_classes = len(
            self.label_to_index
        )

        confusion = torch.zeros(
            (
                num_classes,
                num_classes,
            ),
            dtype=torch.long,
        )

        for true_label, predicted_label in zip(
            y_true,
            y_pred,
        ):

            confusion[
                true_label,
                predicted_label,
            ] += 1

        true_positive = torch.diag(
            confusion
        ).float()

        predicted_positive = (
            confusion.sum(
                dim=0
            ).float()
        )

        actual_positive = (
            confusion.sum(
                dim=1
            ).float()
        )

        precision = torch.where(
            predicted_positive > 0,
            true_positive
            / predicted_positive,
            torch.zeros_like(
                true_positive
            ),
        )

        recall = torch.where(
            actual_positive > 0,
            true_positive
            / actual_positive,
            torch.zeros_like(
                true_positive
            ),
        )

        f1 = torch.where(
            precision + recall > 0,
            2
            * precision
            * recall
            / (
                precision + recall
            ),
            torch.zeros_like(
                precision
            ),
        )

        macro_f1 = float(
            f1.mean().item()
        )

        validation_loss = (
            total_loss
            / total_samples
        )

        return (
            validation_loss,
            accuracy,
            macro_f1,
        )

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int,
    ) -> TrainingHistory:

        for epoch in range(
            1,
            epochs + 1,
        ):

            train_loss = (
                self.train_epoch(
                    train_loader
                )
            )

            (
                validation_loss,
                validation_accuracy,
                validation_f1,
            ) = self.validate(
                validation_loader
            )

            self.history.train_loss.append(
                train_loss
            )

            self.history.validation_loss.append(
                validation_loss
            )

            self.history.validation_accuracy.append(
                validation_accuracy
            )

            self.history.validation_macro_f1.append(
                validation_f1
            )

            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {validation_loss:.4f} | "
                f"Val Acc: {validation_accuracy:.4f} | "
                f"Val F1: {validation_f1:.4f}"
            )

            if (
                validation_f1
                > self.best_validation_f1
            ):

                self.best_validation_f1 = (
                    validation_f1
                )

                save_checkpoint(
                    path=self.checkpoint_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    validation_loss=(
                        validation_loss
                    ),
                    validation_f1=(
                        validation_f1
                    ),
                    model_config=(
                        self.model_config
                    ),
                    training_config=(
                        self.training_config
                    ),
                    label_to_index=(
                        self.label_to_index
                    ),
                )

                print(
                    "  ✓ Best checkpoint saved."
                )

        return self.history

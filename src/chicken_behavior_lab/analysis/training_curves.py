from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True, slots=True)
class TrainingCurveData:
    """
    Structured representation of training history.
    """

    epochs: list[int]
    train_loss: list[float]
    validation_loss: list[float]
    validation_accuracy: list[float]
    validation_macro_f1: list[float]

    def validate(self) -> None:
        """
        Validate training history consistency.
        """

        lengths = {
            len(self.train_loss),
            len(self.validation_loss),
            len(self.validation_accuracy),
            len(self.validation_macro_f1),
        }

        if len(lengths) != 1:
            raise ValueError(
                "All training history arrays "
                "must have the same length."
            )

        if len(self.epochs) != len(
            self.train_loss
        ):
            raise ValueError(
                "epochs and training history "
                "must have the same length."
            )

        if not self.epochs:
            raise ValueError(
                "Training history cannot be empty."
            )


class TrainingHistoryAnalyzer:
    """
    Load and analyze training history.
    """

    def load(
        self,
        path: str | Path,
    ) -> TrainingCurveData:
        """
        Load training history from JSON.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Training history not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data: dict[str, Any] = json.load(
                file
            )

        required_fields = [
            "train_loss",
            "validation_loss",
            "validation_accuracy",
            "validation_macro_f1",
        ]

        for field in required_fields:

            if field not in data:
                raise ValueError(
                    f"Missing training history "
                    f"field: {field}"
                )

        num_epochs = len(
            data["train_loss"]
        )

        history = TrainingCurveData(
            epochs=list(
                range(
                    1,
                    num_epochs + 1,
                )
            ),
            train_loss=[
                float(value)
                for value in data[
                    "train_loss"
                ]
            ],
            validation_loss=[
                float(value)
                for value in data[
                    "validation_loss"
                ]
            ],
            validation_accuracy=[
                float(value)
                for value in data[
                    "validation_accuracy"
                ]
            ],
            validation_macro_f1=[
                float(value)
                for value in data[
                    "validation_macro_f1"
                ]
            ],
        )

        history.validate()

        return history

    def best_epoch(
        self,
        history: TrainingCurveData,
    ) -> int:
        """
        Return the epoch with the highest
        validation Macro-F1.
        """

        history.validate()

        best_index = max(
            range(
                len(
                    history.validation_macro_f1
                )
            ),
            key=lambda index: (
                history.validation_macro_f1[
                    index
                ]
            ),
        )

        return history.epochs[
            best_index
        ]

    def best_validation_f1(
        self,
        history: TrainingCurveData,
    ) -> float:
        """
        Return the best validation Macro-F1.
        """

        history.validate()

        return max(
            history.validation_macro_f1
        )

    def detect_overfitting(
        self,
        history: TrainingCurveData,
        patience: int = 5,
    ) -> bool:
        """
        Detect a simple overfitting pattern.

        The method considers the model potentially
        overfitting when training loss continues
        decreasing while validation loss increases
        for at least `patience` consecutive epochs.
        """

        history.validate()

        if patience < 1:
            raise ValueError(
                "patience must be >= 1."
            )

        if len(history.epochs) <= patience:
            return False

        consecutive = 0

        for index in range(
            1,
            len(history.epochs),
        ):

            train_improving = (
                history.train_loss[index]
                < history.train_loss[index - 1]
            )

            validation_worsening = (
                history.validation_loss[index]
                > history.validation_loss[
                    index - 1
                ]
            )

            if (
                train_improving
                and validation_worsening
            ):
                consecutive += 1

                if consecutive >= patience:
                    return True

            else:
                consecutive = 0

        return False

    def summarize(
        self,
        history: TrainingCurveData,
    ) -> dict[str, Any]:
        """
        Create a compact training summary.
        """

        history.validate()

        best_epoch = self.best_epoch(
            history
        )

        best_f1 = self.best_validation_f1(
            history
        )

        return {
            "num_epochs": len(
                history.epochs
            ),
            "best_epoch": best_epoch,
            "best_validation_macro_f1": (
                best_f1
            ),
            "final_train_loss": (
                history.train_loss[-1]
            ),
            "final_validation_loss": (
                history.validation_loss[-1]
            ),
            "final_validation_accuracy": (
                history.validation_accuracy[-1]
            ),
            "final_validation_macro_f1": (
                history.validation_macro_f1[-1]
            ),
        }

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class TrainingConfig:

    epochs: int = 50

    batch_size: int = 16

    learning_rate: float = 1e-3

    weight_decay: float = 1e-4

    hidden_dim: int = 64

    num_gnn_layers: int = 3

    dropout: float = 0.2

    validation_fraction: float = 0.2

    test_fraction: float = 0.2

    split_group: str = "video"

    random_seed: int = 42

    num_workers: int = 0

    checkpoint_path: str = (
        "checkpoints/"
        "chicken_behavior_gnn_best.pt"
    )

    def validate(self) -> None:

        if self.epochs <= 0:
            raise ValueError(
                "epochs must be > 0."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be > 0."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be > 0."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay must be >= 0."
            )

        if self.hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be > 0."
            )

        if self.num_gnn_layers <= 0:
            raise ValueError(
                "num_gnn_layers must be > 0."
            )

        if not (
            0.0
            <= self.validation_fraction
            < 1.0
        ):
            raise ValueError(
                "validation_fraction must "
                "be in [0, 1)."
            )

        if not (
            0.0
            <= self.test_fraction
            < 1.0
        ):
            raise ValueError(
                "test_fraction must "
                "be in [0, 1)."
            )

        if (
            self.validation_fraction
            + self.test_fraction
            >= 1.0
        ):
            raise ValueError(
                "validation_fraction + "
                "test_fraction must be < 1."
            )

        if self.split_group not in {
            "video",
            "track",
        }:
            raise ValueError(
                "split_group must be "
                "'video' or 'track'."
            )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers must be >= 0."
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to a plain dictionary.
        """

        return asdict(self)

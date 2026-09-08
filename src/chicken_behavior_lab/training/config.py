from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """
    Configuration for behavior-classification training.
    """

    epochs: int = 50

    batch_size: int = 16

    learning_rate: float = 1e-3

    weight_decay: float = 1e-4

    hidden_dim: int = 64

    num_gnn_layers: int = 3

    dropout: float = 0.2

    validation_fraction: float = 0.2

    random_seed: int = 42

    num_workers: int = 0

    checkpoint_path: str = (
        "checkpoints/"
        "chicken_behavior_gnn_best.pt"
    )

    def validate(self) -> None:
        """
        Validate configuration values.
        """

        if self.epochs <= 0:
            raise ValueError(
                "epochs must be positive."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative."
            )

        if self.hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if self.num_gnn_layers <= 0:
            raise ValueError(
                "num_gnn_layers must be positive."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError(
                "validation_fraction must be "
                "between 0 and 1."
            )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers cannot be negative."
            )

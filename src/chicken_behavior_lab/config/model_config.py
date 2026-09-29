from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    """
    Configuration for ChickenBehaviorLab models.

    Supports both:
        - spatial/baseline graph models
        - temporal spatio-temporal graph models
    """

    model_type: str

    node_feature_dim: int
    edge_feature_dim: int | None

    spatial_hidden_dim: int
    temporal_hidden_dim: int

    num_classes: int

    num_gnn_layers: int = 2
    num_gru_layers: int = 1

    dropout: float = 0.0

    bidirectional_gru: bool = False

    sequence_length: int = 1
    sequence_stride: int = 1

    hidden_dim: int | None = None

    def __post_init__(self) -> None:
        if not self.model_type:
            raise ValueError(
                "model_type cannot be empty."
            )

        if self.node_feature_dim < 1:
            raise ValueError(
                "node_feature_dim must be >= 1."
            )

        if (
            self.edge_feature_dim is not None
            and self.edge_feature_dim < 1
        ):
            raise ValueError(
                "edge_feature_dim must be >= 1 or None."
            )

        if self.spatial_hidden_dim < 1:
            raise ValueError(
                "spatial_hidden_dim must be >= 1."
            )

        if self.temporal_hidden_dim < 1:
            raise ValueError(
                "temporal_hidden_dim must be >= 1."
            )

        if self.num_classes < 2:
            raise ValueError(
                "num_classes must be >= 2."
            )

        if self.num_gnn_layers < 1:
            raise ValueError(
                "num_gnn_layers must be >= 1."
            )

        if self.num_gru_layers < 1:
            raise ValueError(
                "num_gru_layers must be >= 1."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        if self.sequence_length < 1:
            raise ValueError(
                "sequence_length must be >= 1."
            )

        if self.sequence_stride < 1:
            raise ValueError(
                "sequence_stride must be >= 1."
            )

        if self.hidden_dim is not None:
            if self.hidden_dim < 1:
                raise ValueError(
                    "hidden_dim must be >= 1 or None."
                )

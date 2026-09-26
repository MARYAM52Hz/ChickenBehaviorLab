from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:

    model_type: str = "baseline"

    node_feature_dim: int = 8

    edge_feature_dim: int = 4

    spatial_hidden_dim: int = 64

    temporal_hidden_dim: int = 128

    num_classes: int = 2

    num_gnn_layers: int = 2

    num_gru_layers: int = 1

    dropout: float = 0.2

    bidirectional_gru: bool = False

    sequence_length: int = 16

    sequence_stride: int = 4

    def validate(self) -> None:

        if self.model_type not in {
            "baseline",
            "temporal",
        }:
            raise ValueError(
                "model_type must be either "
                "'baseline' or 'temporal'."
            )

        if self.node_feature_dim < 1:
            raise ValueError(
                "node_feature_dim must be >= 1."
            )

        if self.edge_feature_dim < 1:
            raise ValueError(
                "edge_feature_dim must be >= 1."
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

    @property
    def is_temporal(self) -> bool:
        return self.model_type == "temporal"

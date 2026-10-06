from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chicken_behavior_lab.models.temporal_behavior_gnn import (
    TemporalBehaviorGNN,
)


@dataclass(slots=True)
class TemporalModelConfig:
    """
    Configuration for TemporalBehaviorGNN.
    """

    node_feature_dim: int

    edge_feature_dim: int | None

    spatial_hidden_dim: int = 64

    temporal_hidden_dim: int = 64

    num_gnn_layers: int = 2

    num_gru_layers: int = 1

    dropout: float = 0.0

    bidirectional_gru: bool = False

    def validate(self) -> None:
        if self.node_feature_dim < 1:
            raise ValueError(
                "node_feature_dim must be >= 1."
            )

        if self.edge_feature_dim is not None:
            if self.edge_feature_dim < 1:
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
                "dropout must satisfy 0.0 <= dropout < 1.0."
            )


def build_temporal_model(
    *,
    model_config: TemporalModelConfig,
    num_classes: int,
) -> TemporalBehaviorGNN:
    """
    Build a TemporalBehaviorGNN using the final number of
    behavior classes discovered from the training split.
    """

    model_config.validate()

    if num_classes < 1:
        raise ValueError(
            "num_classes must be >= 1."
        )

    model = TemporalBehaviorGNN(
        node_feature_dim=(
            model_config.node_feature_dim
        ),
        edge_feature_dim=(
            model_config.edge_feature_dim
        ),
        spatial_hidden_dim=(
            model_config.spatial_hidden_dim
        ),
        temporal_hidden_dim=(
            model_config.temporal_hidden_dim
        ),
        num_classes=num_classes,
        num_gnn_layers=(
            model_config.num_gnn_layers
        ),
        num_gru_layers=(
            model_config.num_gru_layers
        ),
        dropout=model_config.dropout,
        bidirectional_gru=(
            model_config.bidirectional_gru
        ),
    )

    return model


def infer_temporal_model_dimensions(
    dataset,
) -> dict[str, Any]:
    """
    Infer input dimensions from a TemporalPyGDataset.

    This helper inspects the first temporal graph and returns
    dimensions required by TemporalBehaviorGNN.
    """

    if len(dataset) == 0:
        raise ValueError(
            "Cannot infer model dimensions from an empty dataset."
        )

    sample = dataset[0]

    x = sample.x

    if x.ndim != 3:
        raise ValueError(
            "Temporal dataset x must have shape [T, N, F]."
        )

    node_feature_dim = int(
        x.shape[-1]
    )

    edge_feature_dim = None

    if hasattr(
        sample,
        "edge_attr",
    ):
        edge_attr = sample.edge_attr

        if edge_attr is not None:
            if edge_attr.ndim != 3:
                raise ValueError(
                    "Temporal edge_attr must have shape "
                    "[T, E, F_edge]."
                )

            edge_feature_dim = int(
                edge_attr.shape[-1]
            )

    return {
        "node_feature_dim": node_feature_dim,
        "edge_feature_dim": edge_feature_dim,
    }


def build_temporal_model_from_dataset(
    *,
    dataset,
    model_config: TemporalModelConfig | None = None,
) -> TemporalBehaviorGNN:
    """
    Build a TemporalBehaviorGNN directly from a prepared dataset.

    The number of classes is taken from the dataset label mapping.
    """

    dimensions = infer_temporal_model_dimensions(
        dataset
    )

    if model_config is None:
        model_config = TemporalModelConfig(
            node_feature_dim=(
                dimensions["node_feature_dim"]
            ),
            edge_feature_dim=(
                dimensions["edge_feature_dim"]
            ),
        )
    else:
        if (
            model_config.node_feature_dim
            != dimensions["node_feature_dim"]
        ):
            raise ValueError(
                "model_config.node_feature_dim does not "
                "match the dataset."
            )

        if (
            model_config.edge_feature_dim
            != dimensions["edge_feature_dim"]
        ):
            raise ValueError(
                "model_config.edge_feature_dim does not "
                "match the dataset."
            )

    return build_temporal_model(
        model_config=model_config,
        num_classes=len(
            dataset.label_to_index
        ),
    )

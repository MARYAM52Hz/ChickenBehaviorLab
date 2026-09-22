from __future__ import annotations

import torch.nn as nn

from chicken_behavior_lab.config.model_config import (
    ModelConfig,
)

from chicken_behavior_lab.models.chicken_behavior_gnn import (
    ChickenBehaviorGNN,
)

from chicken_behavior_lab.models.temporal_behavior_gnn import (
    TemporalBehaviorGNN,
)


def build_model(
    config: ModelConfig,
) -> nn.Module:
    """
    Build a behavior recognition model from ModelConfig.
    """

    config.validate()

    if config.model_type == "baseline":
        return ChickenBehaviorGNN(
            node_feature_dim=config.node_feature_dim,
            edge_feature_dim=config.edge_feature_dim,
            hidden_dim=config.spatial_hidden_dim,
            num_classes=config.num_classes,
            num_layers=config.num_gnn_layers,
            dropout=config.dropout,
        )

    if config.model_type == "temporal":
        return TemporalBehaviorGNN(
            node_feature_dim=config.node_feature_dim,
            edge_feature_dim=config.edge_feature_dim,
            spatial_hidden_dim=config.spatial_hidden_dim,
            temporal_hidden_dim=config.temporal_hidden_dim,
            num_classes=config.num_classes,
            num_gnn_layers=config.num_gnn_layers,
            num_gru_layers=config.num_gru_layers,
            dropout=config.dropout,
            bidirectional_gru=config.bidirectional_gru,
        )

    raise ValueError(
        f"Unsupported model_type: "
        f"{config.model_type}"
    )

from __future__ import annotations

import torch
from torch import nn

from chicken_behavior_lab.dataset.temporal_batch import (
    TemporalBatch,
)

from chicken_behavior_lab.models.spatial_encoder import (
    SpatialGraphEncoder,
)

from chicken_behavior_lab.models.temporal_encoder import (
    GRUTemporalEncoder,
)


class TemporalBehaviorGNN(nn.Module):
    """
    Spatio-temporal graph model for chicken behavior
    recognition.

    Architecture
    ------------
    Temporal skeleton graphs
        ↓
    Spatial GNN at each time step
        ↓
    Global node pooling
        ↓
    GRU temporal encoder
        ↓
    Classification head
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int | None,
        spatial_hidden_dim: int,
        temporal_hidden_dim: int,
        num_classes: int,
        num_gnn_layers: int = 2,
        num_gru_layers: int = 1,
        dropout: float = 0.0,
        bidirectional_gru: bool = False,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError(
                "num_classes must be >= 2."
            )

        self.spatial_encoder = (
            SpatialGraphEncoder(
                node_feature_dim=(
                    node_feature_dim
                ),
                edge_feature_dim=(
                    edge_feature_dim
                ),
                hidden_dim=(
                    spatial_hidden_dim
                ),
                num_layers=num_gnn_layers,
                dropout=dropout,
            )
        )

        self.temporal_encoder = (
            GRUTemporalEncoder(
                input_dim=spatial_hidden_dim,
                hidden_dim=temporal_hidden_dim,
                num_layers=num_gru_layers,
                dropout=dropout,
                bidirectional=bidirectional_gru,
            )
        )

        temporal_output_dim = (
            self.temporal_encoder.output_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                temporal_output_dim,
                temporal_output_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                temporal_output_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        batch: TemporalBatch,
    ) -> torch.Tensor:
        batch.validate()

        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        temporal_embeddings = []

        for time_index in range(
            sequence_length
        ):
            x_t = x[
                :,
                time_index,
                :,
                :,
            ]

            if edge_attr is not None:
                edge_attr_t = edge_attr[
                    :,
                    time_index,
                    :,
                    :,
                ]
            else:
                edge_attr_t = None

            spatial_embeddings = []

            for batch_index in range(
                batch_size
            ):
                node_embeddings = (
                    self.spatial_encoder(
                        x=x_t[batch_index],
                        edge_index=edge_index,
                        edge_attr=(
                            edge_attr_t[batch_index]
                            if edge_attr_t is not None
                            else None
                        ),
                    )
                )

                pooled = (
                    node_embeddings.mean(
                        dim=0
                    )
                )

                spatial_embeddings.append(
                    pooled
                )

            temporal_embeddings.append(
                torch.stack(
                    spatial_embeddings,
                    dim=0,
                )
            )

        sequence_embedding = torch.stack(
            temporal_embeddings,
            dim=1,
        )

        temporal_output = (
            self.temporal_encoder(
                sequence_embedding
            )
        )

        logits = self.classifier(
            temporal_output
        )

        return logits

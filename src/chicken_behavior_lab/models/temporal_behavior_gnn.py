from __future__ import annotations

import torch
import torch.nn as nn

from chicken_behavior_lab.dataset.temporal_batch import (
    TemporalBatch,
)

from chicken_behavior_lab.models.edge_aware_gnn import (
    EdgeAwareGNN,
)

from chicken_behavior_lab.models.graph_pooling import (
    GraphMeanPooling,
)

from chicken_behavior_lab.models.temporal_encoder import (
    GRUTemporalEncoder,
)


class TemporalBehaviorGNN(nn.Module):
    """
    Temporal graph neural network.

    Pipeline:

        Graph + Edge Features
                ↓
          Edge-aware GNN
                ↓
          Graph Pooling
                ↓
          Temporal Sequence
                ↓
               GRU
                ↓
           Classifier
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        spatial_hidden_dim: int,
        temporal_hidden_dim: int,
        num_classes: int,
        num_gnn_layers: int = 2,
        num_gru_layers: int = 1,
        dropout: float = 0.2,
        bidirectional_gru: bool = False,
    ) -> None:

        super().__init__()

        self.spatial_encoder = EdgeAwareGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=spatial_hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )

        self.pooling = GraphMeanPooling()

        self.temporal_encoder = GRUTemporalEncoder(
            input_dim=spatial_hidden_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=num_gru_layers,
            dropout=dropout,
            bidirectional=bidirectional_gru,
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                self.temporal_encoder.output_dim,
                temporal_hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                temporal_hidden_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        batch: TemporalBatch,
    ) -> torch.Tensor:

        if not isinstance(
            batch,
            TemporalBatch,
        ):
            raise TypeError(
                "TemporalBehaviorGNN expects "
                "a TemporalBatch."
            )

        frame_embeddings = []

        for t in range(
            batch.sequence_length
        ):

            pyg_batch = (
                batch.frame_batches[t]
            )

            x = pyg_batch.x

            edge_index = (
                pyg_batch.edge_index
            )

            edge_attr = (
                pyg_batch.edge_attr
            )

            if edge_attr is None:
                raise ValueError(
                    "TemporalBehaviorGNN requires "
                    "edge_attr."
                )

            node_embeddings = (
                self.spatial_encoder(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                )
            )

            graph_embeddings = (
                self._pool_batched_graphs(
                    node_embeddings,
                    pyg_batch.batch,
                )
            )

            frame_embeddings.append(
                graph_embeddings
            )

        temporal_input = torch.stack(
            frame_embeddings,
            dim=1,
        )

        temporal_embedding = (
            self.temporal_encoder(
                temporal_input
            )
        )

        logits = self.classifier(
            temporal_embedding
        )

        return logits

    @staticmethod
    def _pool_batched_graphs(
        node_embeddings: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:

        num_graphs = int(
            batch_index.max().item()
        ) + 1

        pooled = []

        for graph_index in range(
            num_graphs
        ):

            mask = (
                batch_index
                == graph_index
            )

            graph_nodes = (
                node_embeddings[mask]
            )

            pooled.append(
                graph_nodes.mean(
                    dim=0
                )
            )

        return torch.stack(
            pooled,
            dim=0,
        )

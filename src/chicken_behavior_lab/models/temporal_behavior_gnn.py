from __future__ import annotations

import torch
import torch.nn as nn

from chicken_behavior_lab.dataset.temporal_batch import TemporalBatch
from chicken_behavior_lab.models.edge_aware_gnn import EdgeAwareGNN
from chicken_behavior_lab.models.temporal_encoder import GRUTemporalEncoder


class TemporalBehaviorGNN(nn.Module):
    """Edge-aware spatial GNN followed by a mask-aware GRU classifier."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int | None,
        spatial_hidden_dim: int,
        temporal_hidden_dim: int,
        num_classes: int,
        num_gnn_layers: int = 2,
        num_gru_layers: int = 1,
        dropout: float = 0.2,
        bidirectional_gru: bool = False,
    ) -> None:
        super().__init__()
        if edge_feature_dim is not None and edge_feature_dim < 1:
            raise ValueError("edge_feature_dim must be >= 1 or None.")
        self.edge_feature_dim = edge_feature_dim or 1
        self.spatial_encoder = EdgeAwareGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=self.edge_feature_dim,
            hidden_dim=spatial_hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )
        self.temporal_encoder = GRUTemporalEncoder(
            input_dim=spatial_hidden_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=num_gru_layers,
            dropout=dropout,
            bidirectional=bidirectional_gru,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.temporal_encoder.output_dim, temporal_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden_dim, num_classes),
        )

    def forward(self, batch: TemporalBatch) -> torch.Tensor:
        if not isinstance(batch, TemporalBatch):
            raise TypeError("TemporalBehaviorGNN expects a TemporalBatch.")
        batch.validate()

        frame_embeddings = []
        for t, pyg_batch in enumerate(batch.frame_batches):
            x = pyg_batch.x
            edge_index = pyg_batch.edge_index
            edge_attr = getattr(pyg_batch, "edge_attr", None)
            if edge_attr is None:
                edge_attr = x.new_zeros((edge_index.shape[1], self.edge_feature_dim))
            elif edge_attr.ndim != 2 or edge_attr.shape[1] != self.edge_feature_dim:
                raise ValueError(
                    f"Each frame edge_attr must have shape (E, {self.edge_feature_dim})."
                )

            node_embeddings = self.spatial_encoder(x, edge_index, edge_attr)
            valid = batch.node_mask[:, t, :].reshape(-1)
            node_embeddings = node_embeddings * valid.unsqueeze(-1).to(node_embeddings.dtype)
            pooled = self._masked_pool(node_embeddings, pyg_batch.batch, valid, batch.batch_size)
            frame_embeddings.append(pooled)

        temporal_input = torch.stack(frame_embeddings, dim=1)
        encoded = []
        # Compact valid timesteps per sample. This handles internal gaps as well
        # as right-padding; a packed-sequence prefix assumption is not made.
        for sample_index in range(batch.batch_size):
            valid_steps = batch.frame_mask[sample_index]
            sequence = temporal_input[sample_index:sample_index + 1, valid_steps, :]
            if sequence.shape[1] == 0:
                raise ValueError(f"Sample {batch.sample_id[sample_index]} has no valid frames.")
            encoded.append(self.temporal_encoder(sequence))
        temporal_embedding = torch.cat(encoded, dim=0)
        return self.classifier(temporal_embedding)

    @staticmethod
    def _masked_pool(
        node_embeddings: torch.Tensor,
        batch_index: torch.Tensor,
        valid_mask: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        sums = node_embeddings.new_zeros((num_graphs, node_embeddings.shape[-1]))
        counts = node_embeddings.new_zeros((num_graphs, 1))
        sums.index_add_(0, batch_index, node_embeddings * valid_mask.unsqueeze(-1))
        counts.index_add_(0, batch_index, valid_mask.to(node_embeddings.dtype).unsqueeze(-1))
        return sums / counts.clamp_min(1.0)

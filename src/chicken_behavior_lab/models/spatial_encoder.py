from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINEConv


class SpatialGraphEncoder(nn.Module):
    """
    Encode one skeleton graph into node embeddings.

    Input:
        x:
            [N, F_node]

        edge_index:
            [2, E]

        edge_attr:
            [E, F_edge] or None

    Output:
        [N, H]
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int | None,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if node_feature_dim < 1:
            raise ValueError(
                "node_feature_dim must be >= 1."
            )

        if hidden_dim < 1:
            raise ValueError(
                "hidden_dim must be >= 1."
            )

        if num_layers < 1:
            raise ValueError(
                "num_layers must be >= 1."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        self.node_projection = nn.Linear(
            node_feature_dim,
            hidden_dim,
        )

        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                ),
                nn.ReLU(),
                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                ),
            )

            self.convs.append(
                GINEConv(
                    nn=mlp,
                    edge_dim=edge_feature_dim,
                )
            )

        self.dropout = nn.Dropout(
            dropout
        )

        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                "x must have shape [N, F]."
            )

        if edge_index.ndim != 2:
            raise ValueError(
                "edge_index must have shape [2, E]."
            )

        if edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, E]."
            )

        if edge_attr is not None:
            if edge_attr.ndim != 2:
                raise ValueError(
                    "edge_attr must have shape [E, F_edge]."
                )

            if (
                edge_attr.shape[0]
                != edge_index.shape[1]
            ):
                raise ValueError(
                    "edge_attr and edge_index must "
                    "contain the same number of edges."
                )

        x = self.node_projection(x)

        for conv in self.convs:
            x = conv(
                x,
                edge_index,
                edge_attr,
            )

            x = self.activation(x)
            x = self.dropout(x)

        return x

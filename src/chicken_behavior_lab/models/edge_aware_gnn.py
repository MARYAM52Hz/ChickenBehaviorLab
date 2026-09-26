from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv


class EdgeAwareGNN(nn.Module):
    """
    Graph encoder using edge-aware message passing.

    GINEConv is used because it explicitly incorporates
    edge attributes into the message-passing operation.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:

        super().__init__()

        if num_layers < 1:
            raise ValueError(
                "num_layers must be >= 1."
            )

        self.node_projection = nn.Linear(
            node_feature_dim,
            hidden_dim,
        )

        self.edge_projection = nn.Linear(
            edge_feature_dim,
            hidden_dim,
        )

        self.convs = nn.ModuleList()

        self.norms = nn.ModuleList()

        for _ in range(num_layers):

            mlp = nn.Sequential(
                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                ),
            )

            self.convs.append(
                GINEConv(
                    nn=mlp,
                    edge_dim=hidden_dim,
                )
            )

            self.norms.append(
                nn.LayerNorm(hidden_dim)
            )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:

        x = self.node_projection(x)

        edge_attr = self.edge_projection(
            edge_attr
        )

        for conv, norm in zip(
            self.convs,
            self.norms,
        ):
            residual = x

            x = conv(
                x,
                edge_index,
                edge_attr,
            )

            x = norm(
                x + residual
            )

            x = torch.relu(x)

            x = self.dropout(x)

        return x

from __future__ import annotations

import torch
from torch import Tensor, nn

from torch_geometric.nn import (
    GINEConv,
    global_mean_pool,
)


class ChickenBehaviorGNN(nn.Module):
    """
    Baseline GNN for chicken behavior recognition.

    The model expects a PyTorch Geometric Data object
    containing:

        x
        edge_index
        edge_attr
        batch

    Node features:
        x.shape = (N, F)

    Edge features:
        edge_attr.shape = (E, D)

    Output:
        logits.shape = (B, num_classes)
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:

        super().__init__()

        if node_feature_dim <= 0:
            raise ValueError(
                "node_feature_dim must be positive."
            )

        if edge_feature_dim <= 0:
            raise ValueError(
                "edge_feature_dim must be positive."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than 1."
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        self.node_feature_dim = (
            node_feature_dim
        )

        self.edge_feature_dim = (
            edge_feature_dim
        )

        self.hidden_dim = hidden_dim

        self.num_classes = num_classes

        self.num_layers = num_layers

        self.dropout = dropout

        # =================================================
        # Input projection
        # =================================================

        self.input_projection = nn.Linear(
            node_feature_dim,
            hidden_dim,
        )

        # =================================================
        # GINE layers
        # =================================================

        self.convs = nn.ModuleList()

        self.norms = nn.ModuleList()

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

            conv = GINEConv(
                nn=mlp,
                edge_dim=edge_feature_dim,
            )

            self.convs.append(conv)

            self.norms.append(
                nn.BatchNorm1d(
                    hidden_dim
                )
            )

        # =================================================
        # Classifier
        # =================================================

        classifier_hidden_dim = max(
            hidden_dim // 2,
            1,
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim,
                classifier_hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                classifier_hidden_dim,
                num_classes,
            ),
        )

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Node features.

        edge_index:
            Graph connectivity.

        edge_attr:
            Edge features.

        batch:
            Batch assignment for each node.

            If None, all nodes are assumed to belong
            to one graph.

        Returns
        -------
        Tensor
            Class logits.
        """

        # -------------------------------------------------
        # Validate dimensions
        # -------------------------------------------------

        if x.ndim != 2:
            raise ValueError(
                "x must have shape (N, F)."
            )

        if edge_index.ndim != 2:
            raise ValueError(
                "edge_index must have shape (2, E)."
            )

        if edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape (2, E)."
            )

        if edge_attr.ndim != 2:
            raise ValueError(
                "edge_attr must have shape (E, D)."
            )

        if x.shape[1] != self.node_feature_dim:
            raise ValueError(
                "Unexpected node feature dimension: "
                f"{x.shape[1]} != "
                f"{self.node_feature_dim}"
            )

        if edge_attr.shape[1] != (
            self.edge_feature_dim
        ):
            raise ValueError(
                "Unexpected edge feature dimension: "
                f"{edge_attr.shape[1]} != "
                f"{self.edge_feature_dim}"
            )

        # -------------------------------------------------
        # Create batch vector for single graph
        # -------------------------------------------------

        if batch is None:

            batch = torch.zeros(
                x.shape[0],
                dtype=torch.long,
                device=x.device,
            )

        # =================================================
        # Input projection
        # =================================================

        x = self.input_projection(x)

        # =================================================
        # GINE message passing
        # =================================================

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

            x = norm(x)

            x = torch.relu(x)

            x = x + residual

            x = torch.dropout(
                x,
                p=self.dropout,
                train=self.training,
            )

        # =================================================
        # Global graph pooling
        # =================================================

        graph_embedding = global_mean_pool(
            x,
            batch,
        )

        # =================================================
        # Classification
        # =================================================

        logits = self.classifier(
            graph_embedding
        )

        return logits

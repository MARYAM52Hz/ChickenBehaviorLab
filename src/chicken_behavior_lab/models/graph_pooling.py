from __future__ import annotations

import torch
import torch.nn as nn


class GraphMeanPooling(nn.Module):
    """
    Mean pooling over graph nodes.
    """

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:
            [B, N, H]

        returns:
            [B, H]
        """

        if x.ndim != 3:
            raise ValueError(
                "Expected x with shape [B, N, H]."
            )

        return x.mean(
            dim=1
        )

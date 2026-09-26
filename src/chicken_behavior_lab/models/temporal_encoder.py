from __future__ import annotations

import torch
import torch.nn as nn


class GRUTemporalEncoder(nn.Module):
    """
    GRU-based temporal encoder.

    Input:
        [B, T, H]

    Output:
        [B, H_out]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:

        super().__init__()

        if num_layers < 1:
            raise ValueError(
                "num_layers must be >= 1."
            )

        effective_dropout = (
            dropout
            if num_layers > 1
            else 0.0
        )

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
            bidirectional=bidirectional,
        )

        self.bidirectional = bidirectional

        self.output_dim = (
            hidden_dim * 2
            if bidirectional
            else hidden_dim
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "Expected input shape [B, T, H]."
            )

        _, hidden = self.gru(x)

        if self.bidirectional:

            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]

            return torch.cat(
                [
                    forward_hidden,
                    backward_hidden,
                ],
                dim=-1,
            )

        return hidden[-1]

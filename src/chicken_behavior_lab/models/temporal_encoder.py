from __future__ import annotations

import torch
from torch import nn


class GRUTemporalEncoder(nn.Module):
    """
    Encode a sequence of spatial graph embeddings using GRU.

    Input:
        [B, T, H]

    Output:
        [B, H]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        if input_dim < 1:
            raise ValueError(
                "input_dim must be >= 1."
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

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

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

        output_dim = (
            hidden_dim * 2
            if bidirectional
            else hidden_dim
        )

        self.output_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape [B, T, H]."
            )

        _, hidden = self.gru(x)

        if self.bidirectional:
            forward_hidden = hidden[
                -2
            ]

            backward_hidden = hidden[
                -1
            ]

            output = torch.cat(
                [
                    forward_hidden,
                    backward_hidden,
                ],
                dim=-1,
            )
        else:
            output = hidden[-1]

        return output

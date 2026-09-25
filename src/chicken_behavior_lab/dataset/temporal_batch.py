from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Batch


@dataclass
class TemporalBatch:
    """
    Batch of temporal graph sequences.

    Shapes
    ------
    x:
        [B, T, N, F]

    edge_index:
        List-like structure containing graph connectivity
        for each temporal frame.

    edge_attr:
        List-like structure containing edge features
        for each temporal frame.

    y:
        [B]
    """

    x: torch.Tensor

    edge_index: list[torch.Tensor]

    edge_attr: list[torch.Tensor | None]

    y: torch.Tensor

    sample_id: list[str]

    video_id: list[str]

    track_id: torch.Tensor

    start_frame: torch.Tensor

    end_frame: torch.Tensor

    behavior_id: list[str]

    frame_batches: list[list[Batch]]

    @property
    def batch_size(self) -> int:
        return len(
            self.sample_id
        )

    @property
    def sequence_length(self) -> int:
        if self.x.ndim < 2:
            raise ValueError(
                "TemporalBatch.x must have at least "
                "two dimensions."
            )

        return self.x.shape[1]

    def to(
        self,
        device: torch.device | str,
    ) -> "TemporalBatch":

        self.x = self.x.to(device)

        self.y = self.y.to(device)

        self.track_id = (
            self.track_id.to(device)
        )

        self.start_frame = (
            self.start_frame.to(device)
        )

        self.end_frame = (
            self.end_frame.to(device)
        )

        self.edge_index = [
            edge.to(device)
            for edge in self.edge_index
        ]

        self.edge_attr = [
            None
            if edge is None
            else edge.to(device)
            for edge in self.edge_attr
        ]

        for frame_batches in self.frame_batches:
            for batch in frame_batches:
                batch.to(device)

        return self

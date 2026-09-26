
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Batch


@dataclass
class TemporalBatch:
    """
    Batch of temporal graph sequences.

    x:
        [B, T, N, F]

    frame_batches:
        PyG Batch object for every temporal frame.

    edge_attr:
        One tensor per temporal frame with shape:

            [B, E, D]

        or None when edge features are unavailable.
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

    frame_batches: list[Batch]

    @property
    def batch_size(self) -> int:
        return len(self.sample_id)

    @property
    def sequence_length(self) -> int:
        return self.x.shape[1]

    @property
    def node_feature_dim(self) -> int:
        return self.x.shape[-1]

    def to(
        self,
        device: torch.device | str,
    ) -> "TemporalBatch":

        self.x = self.x.to(device)
        self.y = self.y.to(device)

        self.track_id = self.track_id.to(device)
        self.start_frame = self.start_frame.to(device)
        self.end_frame = self.end_frame.to(device)

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

        self.frame_batches = [
            batch.to(device)
            for batch in self.frame_batches
        ]

        return self

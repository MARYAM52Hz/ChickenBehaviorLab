"""Validated batch representation shared by temporal graph training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Batch


@dataclass
class TemporalBatch:
    """A batch of fixed-length temporal graph sequences.

    Tensor shapes:
        x:          [B, T, N, F]
        edge_index: [2, E] (shared topology)
        edge_attr:  [B, T, E, D] or None
        y:          [B]

    frame_batches[t] is a PyG batch containing one graph per sample
    for temporal step t. The temporal GNN consumes these objects.
    Track identifiers stay as Python strings and are never converted to
    numeric tensors.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor | None
    y: torch.Tensor
    sample_id: list[str]
    video_id: list[str]
    track_id: list[str | None]
    start_frame: torch.Tensor
    end_frame: torch.Tensor
    behavior_id: list[str]
    frame_batches: list[Batch]

    @property
    def batch_size(self) -> int:
        return self.x.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.x.shape[1]

    @property
    def num_nodes(self) -> int:
        return self.x.shape[2]

    @property
    def node_feature_dim(self) -> int:
        return self.x.shape[3]

    def validate(self) -> None:
        """Validate tensor dimensions, graph topology, and metadata lengths."""
        if self.x.ndim != 4:
            raise ValueError("x must have shape (B, T, N, F).")
        batch_size, temporal_length, num_nodes, feature_dim = self.x.shape
        if batch_size < 1 or temporal_length < 1 or num_nodes < 1 or feature_dim < 1:
            raise ValueError("x must contain at least one batch, frame, node, and feature.")

        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E).")
        if self.edge_index.dtype != torch.long:
            raise TypeError("edge_index must use torch.long indices.")
        if self.edge_index.numel() and (
            torch.any(self.edge_index < 0) or torch.any(self.edge_index >= num_nodes)
        ):
            raise ValueError("edge_index contains a node index outside [0, N).")

        if self.edge_attr is not None:
            if self.edge_attr.ndim != 4:
                raise ValueError("edge_attr must have shape (B, T, E, D).")
            if self.edge_attr.shape[:3] != (
                batch_size, temporal_length, self.edge_index.shape[1]
            ):
                raise ValueError("edge_attr dimensions must match (B, T, E).")

        if self.y.shape != (batch_size,):
            raise ValueError("y must have shape (B,).")
        if self.start_frame.shape != (batch_size,) or self.end_frame.shape != (batch_size,):
            raise ValueError("start_frame and end_frame must each have shape (B,).")
        if torch.any(self.start_frame < 0) or torch.any(self.end_frame < self.start_frame):
            raise ValueError("Frame ranges must be non-negative and ordered.")

        metadata = (self.sample_id, self.video_id, self.track_id, self.behavior_id)
        if any(len(values) != batch_size for values in metadata):
            raise ValueError("Each metadata list must contain exactly B values.")
        if any(not isinstance(value, str) or not value for value in self.sample_id):
            raise ValueError("sample_id values must be non-empty strings.")
        if any(not isinstance(value, str) or not value for value in self.video_id):
            raise ValueError("video_id values must be non-empty strings.")
        if any(value is not None and (not isinstance(value, str) or not value) for value in self.track_id):
            raise ValueError("track_id values must be canonical strings or None.")
        if any(not isinstance(value, str) or not value for value in self.behavior_id):
            raise ValueError("behavior_id values must be non-empty strings.")

        if len(self.frame_batches) != temporal_length:
            raise ValueError("frame_batches must contain one PyG batch per temporal step.")
        for time_index, frame_batch in enumerate(self.frame_batches):
            if frame_batch.num_graphs != batch_size:
                raise ValueError(f"frame_batches[{time_index}] must contain B graphs.")
            if frame_batch.x.ndim != 2 or frame_batch.x.shape != (
                batch_size * num_nodes, feature_dim
            ):
                raise ValueError(f"frame_batches[{time_index}].x has an incompatible shape.")

    def to(self, device: torch.device | str) -> "TemporalBatch":
        """Move tensor payloads and graph batches to a device; keep IDs as strings."""
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.y = self.y.to(device)
        self.start_frame = self.start_frame.to(device)
        self.end_frame = self.end_frame.to(device)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device)
        self.frame_batches = [frame_batch.to(device) for frame_batch in self.frame_batches]
        return self

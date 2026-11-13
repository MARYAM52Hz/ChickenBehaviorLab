"""Validated padded batch representation for annotation-level temporal graphs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Batch


@dataclass
class TemporalBatch:
    """Batch tensors use x=[B,T,N,F], with explicit validity masks."""

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
    frame_mask: torch.Tensor
    node_mask: torch.Tensor
    edge_mask: torch.Tensor

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
        if self.x.ndim != 4:
            raise ValueError("x must have shape (B, T, N, F).")
        b, t, n, f = self.x.shape
        if min(b, t, n, f) < 1:
            raise ValueError("x dimensions must all be positive.")
        if self.frame_mask.shape != (b, t):
            raise ValueError("frame_mask must have shape (B, T).")
        if self.node_mask.shape != (b, t, n):
            raise ValueError("node_mask must have shape (B, T, N).")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E).")
        e = self.edge_index.shape[1]
        if self.edge_mask.shape != (b, t, e):
            raise ValueError("edge_mask must have shape (B, T, E).")
        if self.edge_index.dtype != torch.long:
            raise TypeError("edge_index must use torch.long.")
        if self.edge_index.numel() and (
            torch.any(self.edge_index < 0) or torch.any(self.edge_index >= n)
        ):
            raise ValueError("edge_index contains an invalid node index.")
        if self.edge_attr is not None:
            if self.edge_attr.ndim != 4 or self.edge_attr.shape[:3] != (b, t, e):
                raise ValueError("edge_attr must have shape (B, T, E, D).")
        if self.y.shape != (b,):
            raise ValueError("y must have shape (B,).")
        if self.start_frame.shape != (b,) or self.end_frame.shape != (b,):
            raise ValueError("Frame metadata tensors must have shape (B,).")
        if torch.any(self.start_frame < 0) or torch.any(self.end_frame < self.start_frame):
            raise ValueError("Frame ranges must be non-negative and ordered.")
        metadata = (self.sample_id, self.video_id, self.track_id, self.behavior_id)
        if any(len(values) != b for values in metadata):
            raise ValueError("Each metadata list must contain exactly B values.")
        if any(not isinstance(value, str) or not value for value in self.sample_id):
            raise ValueError("sample_id values must be non-empty strings.")
        if any(not isinstance(value, str) or not value for value in self.video_id):
            raise ValueError("video_id values must be non-empty strings.")
        if any(value is not None and (not isinstance(value, str) or not value) for value in self.track_id):
            raise ValueError("track_id values must be non-empty strings or None.")
        if any(not isinstance(value, str) or not value for value in self.behavior_id):
            raise ValueError("behavior_id values must be non-empty strings.")
        if torch.any(~self.frame_mask.any(dim=1)):
            raise ValueError("Every sample must contain at least one valid frame.")
        if torch.any(~(self.node_mask & self.frame_mask.unsqueeze(-1)).any(dim=(1, 2))):
            raise ValueError("Every sample must contain at least one valid node.")
        if len(self.frame_batches) != t:
            raise ValueError("frame_batches must contain one PyG batch per padded timestep.")
        for index, frame_batch in enumerate(self.frame_batches):
            if frame_batch.num_graphs != b:
                raise ValueError(f"frame_batches[{index}] must contain B graphs.")
            if frame_batch.x.ndim != 2 or frame_batch.x.shape != (b * n, f):
                raise ValueError(f"frame_batches[{index}].x has an incompatible shape.")

    def to(self, device: torch.device | str) -> "TemporalBatch":
        for name in ("x", "edge_index", "y", "start_frame", "end_frame",
                     "frame_mask", "node_mask", "edge_mask"):
            setattr(self, name, getattr(self, name).to(device))
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device)
        self.frame_batches = [item.to(device) for item in self.frame_batches]
        return self

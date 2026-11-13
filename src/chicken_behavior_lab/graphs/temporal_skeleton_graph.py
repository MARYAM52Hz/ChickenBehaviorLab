from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chicken_behavior_lab.core.enums.keypoints import KeypointType


@dataclass(slots=True)
class TemporalSkeletonGraph:
    """Temporal skeleton graph with explicit frame and node validity masks."""

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    frame_ids: tuple[str, ...]
    frame_mask: np.ndarray | None = None
    node_mask: np.ndarray | None = None
    keypoint_types: tuple[KeypointType, ...] | None = None

    def validate(self) -> None:
        if self.node_features.ndim != 3:
            raise ValueError("node_features must have shape (T, V, F).")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E).")
        if self.edge_features.ndim != 3:
            raise ValueError("edge_features must have shape (T, E, D).")

        num_frames, num_nodes, _ = self.node_features.shape
        num_edges = self.edge_index.shape[1]
        if self.edge_features.shape[:2] != (num_frames, num_edges):
            raise ValueError("edge_features must have shape (T, E, D).")
        if len(self.frame_ids) != num_frames:
            raise ValueError("frame_ids length must match the number of frames.")
        if self.frame_mask is not None and self.frame_mask.shape != (num_frames,):
            raise ValueError("frame_mask must have shape (T,).")
        if self.node_mask is not None and self.node_mask.shape != (num_frames, num_nodes):
            raise ValueError("node_mask must have shape (T, V).")
        if self.keypoint_types is not None:
            if len(self.keypoint_types) != num_nodes:
                raise ValueError("keypoint_types length must match the number of nodes.")
            if len(set(self.keypoint_types)) != num_nodes:
                raise ValueError("keypoint_types must contain unique values.")
        if num_edges and (np.any(self.edge_index < 0) or np.any(self.edge_index >= num_nodes)):
            raise ValueError("edge_index contains an invalid node.")

    @property
    def num_frames(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[1]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

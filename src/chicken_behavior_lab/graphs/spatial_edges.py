from __future__ import annotations

import numpy as np

from chicken_behavior_lab.skeleton.topology import canonicalize_edge_index


class SpatialEdgeExpander:
    """Expand canonical anatomical connections over frames as bidirectional edges."""

    def build(
        self,
        skeleton_edge_index: np.ndarray,
        num_frames: int,
        num_keypoints: int,
    ) -> np.ndarray:
        if num_frames <= 0 or num_keypoints <= 0:
            raise ValueError("num_frames and num_keypoints must be positive.")
        base = canonicalize_edge_index(
            skeleton_edge_index,
            num_keypoints=num_keypoints,
            bidirectional=True,
        )
        if base.shape[1] == 0:
            return np.empty((2, 0), dtype=np.int64)

        expanded = [
            base + frame_index * num_keypoints
            for frame_index in range(num_frames)
        ]
        return np.concatenate(expanded, axis=1)

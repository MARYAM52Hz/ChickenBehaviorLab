from __future__ import annotations

import numpy as np


class SpatialEdgeExpander:
    """Expand canonical anatomical connections over frames as bidirectional edges."""

    def build(
        self,
        skeleton_edge_index: np.ndarray,
        num_frames: int,
        num_keypoints: int,
    ) -> np.ndarray:
        edges = np.asarray(skeleton_edge_index, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("skeleton_edge_index must have shape (2, E).")
        if num_frames <= 0 or num_keypoints <= 0:
            raise ValueError("num_frames and num_keypoints must be positive.")
        if edges.size and (np.any(edges < 0) or np.any(edges >= num_keypoints)):
            raise ValueError("Skeleton edge index is outside the keypoint range.")

        canonical: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for source, target in edges.T.tolist():
            if source == target:
                raise ValueError("Skeleton topology cannot contain self-loops.")
            for edge in ((source, target), (target, source)):
                if edge not in seen:
                    canonical.append(edge)
                    seen.add(edge)

        base = np.asarray(canonical, dtype=np.int64).T
        if not canonical:
            return np.empty((2, 0), dtype=np.int64)
        expanded = [
            base + frame_index * num_keypoints
            for frame_index in range(num_frames)
        ]
        return np.concatenate(expanded, axis=1)

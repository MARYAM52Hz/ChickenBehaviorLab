"""Canonical skeleton topology utilities shared by graph representations."""

from __future__ import annotations

import numpy as np


def canonicalize_edge_index(
    edge_index: np.ndarray,
    *,
    num_keypoints: int | None = None,
    bidirectional: bool = True,
) -> np.ndarray:
    """Validate and deduplicate skeleton edges with stable bidirectional ordering.

    Input edges are retained in order. When bidirectional is enabled, each
    reverse edge is appended immediately after its forward edge unless it
    already exists in the input. Self-loops are rejected.
    """
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E).")
    if edges.size:
        if np.any(edges < 0):
            raise ValueError("edge indices cannot be negative.")
        if num_keypoints is not None and np.any(edges >= num_keypoints):
            raise ValueError("edge index is outside the keypoint range.")

    result: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for source, target in edges.T.tolist():
        if source == target:
            raise ValueError("Skeleton topology cannot contain self-loops.")
        candidates = [(source, target)]
        if bidirectional:
            candidates.append((target, source))
        for edge in candidates:
            if edge not in seen:
                result.append(edge)
                seen.add(edge)

    if not result:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(result, dtype=np.int64).T

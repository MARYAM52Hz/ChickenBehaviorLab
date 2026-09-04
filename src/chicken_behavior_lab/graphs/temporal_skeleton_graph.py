from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TemporalSkeletonGraph:
    """
    Temporal skeleton graph representation.

    Parameters
    ----------
    node_features:
        Array containing node features.

        Shape:
            (T, V, F)

        T = number of frames
        V = number of keypoints
        F = number of node features

    edge_index:
        Graph connectivity.

        Shape:
            (2, E)

        E = number of edges

    edge_features:
        Edge attributes.

        Shape:
            (T, E, D)

        D = number of edge features

    frame_ids:
        Frame identifiers.
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    frame_ids: tuple[str, ...]

    def validate(self) -> None:
        """Validate graph dimensions."""

        if self.node_features.ndim != 3:
            raise ValueError(
                "node_features must have shape "
                "(T, V, F)."
            )

        if self.edge_index.ndim != 2:
            raise ValueError(
                "edge_index must have shape "
                "(2, E)."
            )

        if self.edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape "
                "(2, E)."
            )

        if self.edge_features.ndim != 3:
            raise ValueError(
                "edge_features must have shape "
                "(T, E, D)."
            )

        num_frames = (
            self.node_features.shape[0]
        )

        num_edges = (
            self.edge_index.shape[1]
        )

        if self.edge_features.shape[0] != num_frames:
            raise ValueError(
                "edge_features and node_features "
                "must contain the same number of frames."
            )

        if self.edge_features.shape[1] != num_edges:
            raise ValueError(
                "edge_features and edge_index "
                "must contain the same number of edges."
            )

        if len(self.frame_ids) != num_frames:
            raise ValueError(
                "frame_ids length must match "
                "the number of frames."
            )

    @property
    def num_frames(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[1]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

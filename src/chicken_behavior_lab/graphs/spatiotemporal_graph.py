from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SpatioTemporalGraph:
    """
    Flattened spatio-temporal skeleton graph.

    Each node represents one keypoint at one frame.

    Node indexing
    -------------

    Global node index:

        global_index = frame_index * num_keypoints + keypoint_index

    Example
    -------

    For:

        T = 3 frames
        V = 4 keypoints

    flattened nodes are:

        frame 0: 0, 1, 2, 3
        frame 1: 4, 5, 6, 7
        frame 2: 8, 9, 10, 11


    Attributes
    ----------

    node_features:
        Shape:

            (T * V, F)

    edge_index:
        Shape:

            (2, E_total)

    edge_features:
        Shape:

            (E_total, D)

    edge_type:
        Shape:

            (E_total,)

        0 = spatial edge
        1 = temporal edge

    frame_ids:
        Original frame identifiers.

    num_frames:
        Number of frames.

    num_keypoints:
        Number of keypoints per frame.
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    edge_type: np.ndarray

    frame_ids: tuple[str, ...]

    num_frames: int
    num_keypoints: int

    def validate(self) -> None:
        """
        Validate graph structure.
        """

        if self.node_features.ndim != 2:
            raise ValueError(
                "node_features must have shape "
                "(N, F)."
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

        if self.edge_features.ndim != 2:
            raise ValueError(
                "edge_features must have shape "
                "(E, D)."
            )

        if self.edge_type.ndim != 1:
            raise ValueError(
                "edge_type must have shape (E,)."
            )

        num_nodes = self.node_features.shape[0]

        expected_nodes = (
            self.num_frames
            * self.num_keypoints
        )

        if num_nodes != expected_nodes:
            raise ValueError(
                "node_features contains an unexpected "
                "number of nodes."
            )

        num_edges = self.edge_index.shape[1]

        if self.edge_features.shape[0] != num_edges:
            raise ValueError(
                "edge_features must contain one row "
                "per graph edge."
            )

        if self.edge_type.shape[0] != num_edges:
            raise ValueError(
                "edge_type must contain one value "
                "per graph edge."
            )

        if len(self.frame_ids) != self.num_frames:
            raise ValueError(
                "frame_ids length must match "
                "num_frames."
            )

        if num_edges > 0:
            if np.any(self.edge_index < 0):
                raise ValueError(
                    "edge indices cannot be negative."
                )

            if np.any(
                self.edge_index >= num_nodes
            ):
                raise ValueError(
                    "edge_index contains an invalid node."
                )

        allowed_edge_types = {
            0,
            1,
        }

        if not set(
            np.unique(self.edge_type).tolist()
        ).issubset(allowed_edge_types):
            raise ValueError(
                "edge_type must contain only "
                "0 (spatial) or 1 (temporal)."
            )

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def num_spatial_edges(self) -> int:
        return int(
            np.sum(
                self.edge_type == 0
            )
        )

    @property
    def num_temporal_edges(self) -> int:
        return int(
            np.sum(
                self.edge_type == 1
            )
        )

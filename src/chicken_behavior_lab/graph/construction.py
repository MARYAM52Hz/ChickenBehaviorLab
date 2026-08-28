"""
ChickenBehaviorLab Graph Construction
=====================================

Utilities for converting skeleton topology and temporal
features into graph representations.
"""

from __future__ import annotations

import numpy as np

from chicken_behavior_lab.features.temporal_features import (
    TemporalFeatureSequence,
)

from chicken_behavior_lab.graph.graph import (
    TemporalSkeletonGraph,
)


class SkeletonGraphBuilder:
    """
    Build temporal skeleton graphs from canonical
    skeleton connections.
    """

    def __init__(
        self,
        skeleton_connections: list[
            tuple[int, int]
        ] | tuple[
            tuple[int, int], ...
        ],
        directed: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        skeleton_connections:
            Canonical anatomical connections.

        directed:
            Whether graph edges should remain directed.

            False:
                Each anatomical connection is represented
                in both directions.

            True:
                Connections are kept as provided.
        """

        if not skeleton_connections:

            raise ValueError(
                "skeleton_connections "
                "cannot be empty."
            )

        self.skeleton_connections = tuple(
            skeleton_connections
        )

        self.directed = directed

    # =====================================================
    # Edge construction
    # =====================================================

    def build_edge_index(
        self,
    ) -> np.ndarray:
        """
        Convert skeleton connections into edge_index.

        Returns
        -------
        np.ndarray
            Shape:

                (2, E)
        """

        edges: list[
            tuple[int, int]
        ] = []

        for source, target in (
            self.skeleton_connections
        ):

            if source < 0 or target < 0:

                raise ValueError(
                    "Skeleton node indices "
                    "must be non-negative."
                )

            edges.append(
                (
                    source,
                    target,
                )
            )

            if not self.directed:

                edges.append(
                    (
                        target,
                        source,
                    )
                )

        return np.asarray(
            edges,
            dtype=np.int64,
        ).T

    # =====================================================
    # Temporal graph
    # =====================================================

    def build(
        self,
        sequence: TemporalFeatureSequence,
    ) -> TemporalSkeletonGraph:
        """
        Build a temporal skeleton graph.

        Parameters
        ----------
        sequence:
            Temporal feature sequence.

        Returns
        -------
        TemporalSkeletonGraph
        """

        edge_index = (
            self.build_edge_index()
        )

        num_nodes = (
            sequence.num_keypoints
        )

        # -------------------------------------------------
        # Validate edge indices
        # -------------------------------------------------

        if edge_index.size > 0:

            if np.any(
                edge_index >= num_nodes
            ):

                raise ValueError(
                    "Skeleton connection refers "
                    "to a keypoint outside the "
                    "feature sequence."
                )

        # -------------------------------------------------
        # Build graph
        # -------------------------------------------------

        graph = TemporalSkeletonGraph(
            node_features=(
                sequence.features.copy()
            ),
            edge_index=edge_index,
            frame_mask=(
                sequence.frame_mask.copy()
            ),
            node_mask=(
                sequence.keypoint_mask.copy()
            ),
        )

        graph.validate()

        return graph

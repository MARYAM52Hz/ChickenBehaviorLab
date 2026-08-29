"""
ChickenBehaviorLab Graph Construction
=====================================

Utilities for converting skeleton topology and temporal
features into temporal graph representations.
"""

from __future__ import annotations

import numpy as np

from chicken_behavior_lab.features.temporal_features import (
    TemporalFeatureSequence,
)

from chicken_behavior_lab.graph.edge_features import (
    EdgeFeatureExtractor,
)

from chicken_behavior_lab.graph.graph import (
    TemporalSkeletonGraph,
)


class SkeletonGraphBuilder:
    """
    Build temporal skeleton graphs from canonical
    skeleton connections.

    The builder is responsible for:

        1. Creating graph connectivity.
        2. Creating edge features.
        3. Attaching node features.
        4. Preserving temporal and node masks.
    """

    def __init__(
        self,
        skeleton_connections: list[
            tuple[int, int]
        ]
        | tuple[
            tuple[int, int], ...
        ],
        directed: bool = False,
        edge_feature_extractor: (
            EdgeFeatureExtractor | None
        ) = None,
    ) -> None:
        """
        Parameters
        ----------
        skeleton_connections:
            Canonical anatomical skeleton connections.

        directed:
            Whether graph edges should remain directed.

            False:
                Each anatomical connection is stored
                in both directions.

            True:
                Connections are kept as provided.

        edge_feature_extractor:
            Optional EdgeFeatureExtractor instance.
        """

        # -------------------------------------------------
        # Validate skeleton connections
        # -------------------------------------------------

        if not skeleton_connections:

            raise ValueError(
                "skeleton_connections "
                "cannot be empty."
            )

        self.skeleton_connections = tuple(
            skeleton_connections
        )

        self.directed = directed

        # -------------------------------------------------
        # Edge feature extractor
        # -------------------------------------------------

        self.edge_feature_extractor = (
            edge_feature_extractor
            or EdgeFeatureExtractor()
        )

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

            # -------------------------------------------------
            # Validate source
            # -------------------------------------------------

            if source < 0:

                raise ValueError(
                    "Skeleton node indices "
                    "must be non-negative."
                )

            # -------------------------------------------------
            # Validate target
            # -------------------------------------------------

            if target < 0:

                raise ValueError(
                    "Skeleton node indices "
                    "must be non-negative."
                )

            # -------------------------------------------------
            # Forward edge
            # -------------------------------------------------

            edges.append(
                (
                    source,
                    target,
                )
            )

            # -------------------------------------------------
            # Reverse edge
            # -------------------------------------------------

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
    # Temporal graph construction
    # =====================================================

    def build(
        self,
        sequence: TemporalFeatureSequence,
    ) -> TemporalSkeletonGraph:
        """
        Build a complete temporal skeleton graph.

        Parameters
        ----------
        sequence:
            Temporal feature sequence.

        Returns
        -------
        TemporalSkeletonGraph

        Output:

            node_features:
                T × V × F

            edge_index:
                2 × E

            edge_features:
                T × E × D

            frame_mask:
                T

            node_mask:
                T × V
        """

        # -------------------------------------------------
        # Create graph topology
        # -------------------------------------------------

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
        # Extract edge features
        # -------------------------------------------------

        edge_features = (
            self.edge_feature_extractor.extract(
                node_features=sequence.features,
                edge_index=edge_index,
            )
        )

        # -------------------------------------------------
        # Create temporal graph
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
            edge_features=edge_features,
        )

        # -------------------------------------------------
        # Validate final graph
        # -------------------------------------------------

        graph.validate()

        return graph

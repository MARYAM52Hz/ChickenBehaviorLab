"""
ChickenBehaviorLab Graph Representation
========================================

Core graph data structures for skeleton-based
chicken behavior recognition.

A chicken skeleton is represented as:

    Nodes  -> anatomical keypoints
    Edges  -> anatomical connections

Node features are stored independently from the
graph topology so that the same graph definition
can be used with different feature representations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SkeletonGraph:
    """
    Graph representation of a single chicken skeleton.

    Parameters
    ----------
    node_features:
        Node feature matrix.

        Shape:

            (V, F)

        where:

            V = number of keypoints
            F = feature dimension

    edge_index:
        Graph connectivity.

        Shape:

            (2, E)

        where:

            E = number of directed edges.

    edge_features:
        Optional edge feature matrix.

        Shape:

            (E, D)

    node_mask:
        Validity mask for nodes.

        Shape:

            (V,)

    """

    node_features: np.ndarray

    edge_index: np.ndarray

    edge_features: np.ndarray | None = None

    node_mask: np.ndarray | None = None

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes."""

        return self.node_features.shape[0]

    @property
    def num_node_features(self) -> int:
        """Return node feature dimension."""

        return self.node_features.shape[1]

    @property
    def num_edges(self) -> int:
        """Return the number of edges."""

        return self.edge_index.shape[1]

    @property
    def edge_feature_dimension(self) -> int:
        """Return edge feature dimension."""

        if self.edge_features is None:
            return 0

        return self.edge_features.shape[1]

    def validate(self) -> None:
        """
        Validate graph dimensions and connectivity.
        """

        if self.node_features.ndim != 2:

            raise ValueError(
                "node_features must have shape "
                "(V, F)."
            )

        if self.edge_index.ndim != 2:

            raise ValueError(
                "edge_index must have shape "
                "(2, E)."
            )

        if self.edge_index.shape[0] != 2:

            raise ValueError(
                "edge_index first dimension "
                "must be 2."
            )

        if self.edge_features is not None:

            if self.edge_features.ndim != 2:

                raise ValueError(
                    "edge_features must have "
                    "shape (E, D)."
                )

            if (
                self.edge_features.shape[0]
                != self.num_edges
            ):

                raise ValueError(
                    "Number of edge features "
                    "must equal number of edges."
                )

        if self.node_mask is not None:

            if self.node_mask.shape != (
                self.num_nodes,
            ):

                raise ValueError(
                    "node_mask must have shape "
                    "(V,)."
                )

        if self.num_edges > 0:

            if np.any(
                self.edge_index < 0
            ):

                raise ValueError(
                    "edge_index cannot contain "
                    "negative node indices."
                )

            if np.any(
                self.edge_index
                >= self.num_nodes
            ):

                raise ValueError(
                    "edge_index contains node "
                    "indices outside graph range."
                )


@dataclass(slots=True)
class TemporalSkeletonGraph:
    """
    Temporal sequence of skeleton graphs.

    Parameters
    ----------
    node_features:
        Temporal node feature tensor.

        Shape:

            (T, V, F)

    edge_index:
        Shared spatial graph topology.

        Shape:

            (2, E)

    frame_mask:
        Valid temporal frame mask.

        Shape:

            (T,)

    node_mask:
        Valid node mask.

        Shape:

            (T, V)

    edge_features:
        Optional temporal edge features.

        Shape:

            (T, E, D)
    """

    node_features: np.ndarray

    edge_index: np.ndarray

    frame_mask: np.ndarray

    node_mask: np.ndarray

    edge_features: np.ndarray | None = None

    @property
    def temporal_length(self) -> int:
        """Return number of temporal frames."""

        return self.node_features.shape[0]

    @property
    def num_nodes(self) -> int:
        """Return number of skeleton nodes."""

        return self.node_features.shape[1]

    @property
    def num_node_features(self) -> int:
        """Return node feature dimension."""

        return self.node_features.shape[2]

    @property
    def num_edges(self) -> int:
        """Return number of graph edges."""

        return self.edge_index.shape[1]

    def validate(self) -> None:
        """
        Validate temporal graph dimensions.
        """

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
                "edge_index first dimension "
                "must be 2."
            )

        temporal_length = (
            self.node_features.shape[0]
        )

        num_nodes = (
            self.node_features.shape[1]
        )

        if self.frame_mask.shape != (
            temporal_length,
        ):

            raise ValueError(
                "frame_mask must have shape "
                "(T,)."
            )

        if self.node_mask.shape != (
            temporal_length,
            num_nodes,
        ):

            raise ValueError(
                "node_mask must have shape "
                "(T, V)."
            )

        if self.edge_features is not None:

            if self.edge_features.ndim != 3:

                raise ValueError(
                    "edge_features must have "
                    "shape (T, E, D)."
                )

            if (
                self.edge_features.shape[0]
                != temporal_length
            ):

                raise ValueError(
                    "edge_features temporal "
                    "dimension must equal T."
                )

            if (
                self.edge_features.shape[1]
                != self.num_edges
            ):

                raise ValueError(
                    "edge_features edge "
                    "dimension must equal E."
                )

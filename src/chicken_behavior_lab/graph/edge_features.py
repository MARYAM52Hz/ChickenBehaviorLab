"""
ChickenBehaviorLab Edge Features
=================================

Geometric edge features for skeleton-based
chicken behavior recognition.

For every skeleton edge:

    source -> target

the following features can be computed:

    relative_dx
    relative_dy
    distance

The implementation supports temporal feature tensors
with shape:

    T × V × F
"""

from __future__ import annotations

import numpy as np


class EdgeFeatureExtractor:
    """
    Extract geometric features for skeleton edges.

    Parameters
    ----------
    x_index:
        Index of x coordinate in node features.

    y_index:
        Index of y coordinate in node features.

    epsilon:
        Small numerical stability constant.
    """

    def __init__(
        self,
        x_index: int = 0,
        y_index: int = 1,
        epsilon: float = 1e-6,
    ) -> None:

        self.x_index = x_index
        self.y_index = y_index
        self.epsilon = epsilon

    # =====================================================
    # Main extraction
    # =====================================================

    def extract(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
    ) -> np.ndarray:
        """
        Extract geometric edge features.

        Parameters
        ----------
        node_features:
            Node feature tensor.

            Shape:

                (T, V, F)

        edge_index:
            Graph connectivity.

            Shape:

                (2, E)

        Returns
        -------
        np.ndarray
            Edge feature tensor:

                (T, E, 3)

            Feature order:

                [relative_dx,
                 relative_dy,
                 distance]
        """

        self._validate_inputs(
            node_features,
            edge_index,
        )

        source_indices = (
            edge_index[0]
        )

        target_indices = (
            edge_index[1]
        )

        source_xy = node_features[
            :,
            source_indices,
            [
                self.x_index,
                self.y_index,
            ],
        ]

        target_xy = node_features[
            :,
            target_indices,
            [
                self.x_index,
                self.y_index,
            ],
        ]

        relative = (
            target_xy
            - source_xy
        )

        distance = np.linalg.norm(
            relative,
            axis=-1,
            keepdims=True,
        )

        edge_features = np.concatenate(
            [
                relative,
                distance,
            ],
            axis=-1,
        )

        return edge_features.astype(
            np.float32
        )

    # =====================================================
    # Input validation
    # =====================================================

    def _validate_inputs(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
    ) -> None:
        """
        Validate input dimensions.
        """

        if node_features.ndim != 3:

            raise ValueError(
                "node_features must have shape "
                "(T, V, F)."
            )

        if edge_index.ndim != 2:

            raise ValueError(
                "edge_index must have shape "
                "(2, E)."
            )

        if edge_index.shape[0] != 2:

            raise ValueError(
                "edge_index first dimension "
                "must be 2."
            )

        num_nodes = (
            node_features.shape[1]
        )

        if edge_index.size > 0:

            if np.any(
                edge_index < 0
            ):

                raise ValueError(
                    "edge_index cannot contain "
                    "negative indices."
                )

            if np.any(
                edge_index >= num_nodes
            ):

                raise ValueError(
                    "edge_index contains a "
                    "node index outside the "
                    "node feature range."
                )

        feature_dimension = (
            node_features.shape[2]
        )

        if (
            self.x_index < 0
            or self.x_index >= feature_dimension
        ):

            raise ValueError(
                "x_index is outside the "
                "node feature dimension."
            )

        if (
            self.y_index < 0
            or self.y_index >= feature_dimension
        ):

            raise ValueError(
                "y_index is outside the "
                "node feature dimension."
            )

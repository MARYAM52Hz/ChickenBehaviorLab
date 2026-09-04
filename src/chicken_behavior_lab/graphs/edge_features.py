from __future__ import annotations

import numpy as np


class EdgeFeatureBuilder:
    """
    Build edge features from temporal skeleton node features.

    The initial implementation uses the difference between
    connected keypoints.

    For an edge:

        source → target

    the edge feature is:

        target - source
    """

    def build(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
    ) -> np.ndarray:
        """
        Build edge features.

        Parameters
        ----------
        node_features:
            Shape:

                (T, V, F)

        edge_index:
            Shape:

                (2, E)

        Returns
        -------
        np.ndarray
            Edge features with shape:

                (T, E, F)
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
                "edge_index must have shape "
                "(2, E)."
            )

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        if np.any(
            source_nodes < 0
        ) or np.any(
            target_nodes < 0
        ):
            raise ValueError(
                "edge indices cannot be negative."
            )

        num_nodes = node_features.shape[1]

        if np.any(
            source_nodes >= num_nodes
        ) or np.any(
            target_nodes >= num_nodes
        ):
            raise ValueError(
                "edge index contains invalid "
                "node indices."
            )

        source_features = node_features[
            :,
            source_nodes,
            :,
        ]

        target_features = node_features[
            :,
            target_nodes,
            :,
        ]

        edge_features = (
            target_features
            - source_features
        )

        return edge_features

"""Shared, named edge-feature extraction for temporal skeleton graphs."""

from __future__ import annotations

import numpy as np

from chicken_behavior_lab.features.edge_contract import (
    build_combined_edge_features,
)


class EdgeFeatureExtractor:
    """Extract the canonical nine-channel combined edge representation."""

    def __init__(self, x_index: int = 0, y_index: int = 1, epsilon: float = 1e-6) -> None:
        self.x_index = x_index
        self.y_index = y_index
        self.epsilon = epsilon

    def extract(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        feature_names: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        nodes = np.asarray(node_features)
        if nodes.ndim != 3:
            raise ValueError("node_features must have shape (T, V, F).")
        if feature_names is None and (self.x_index, self.y_index) != (0, 1):
            feature_dim = nodes.shape[-1]
            if not (0 <= self.x_index < feature_dim and 0 <= self.y_index < feature_dim):
                raise ValueError("Coordinate feature indices are outside the node feature dimension.")
            if self.x_index == self.y_index:
                raise ValueError("x_index and y_index must be different.")
            names = [f"feature_{index}" for index in range(feature_dim)]
            names[self.x_index] = "x"
            names[self.y_index] = "y"
            feature_names = tuple(names)
        return build_combined_edge_features(nodes, edge_index, feature_names)

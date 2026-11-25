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
        if feature_names is None and nodes.shape[-1] == 2 and (self.x_index, self.y_index) != (0, 1):
            names = [f"feature_{i}" for i in range(nodes.shape[-1])]
            names[self.x_index], names[self.y_index] = "x", "y"
            feature_names = tuple(names)
        elif feature_names is None and nodes.shape[-1] != 2 and (self.x_index, self.y_index) != (0, 1):
            names = ["x", "y", "vx", "vy", "ax", "ay", "confidence"][:nodes.shape[-1]]
            names[self.x_index], names[self.y_index] = "x", "y"
            feature_names = tuple(names)
        return build_combined_edge_features(nodes, edge_index, feature_names)

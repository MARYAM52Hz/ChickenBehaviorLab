"""Compatibility facade for the shared combined edge-feature contract."""

from __future__ import annotations

import numpy as np

from chicken_behavior_lab.features.edge_contract import build_combined_edge_features


class EdgeFeatureBuilder:
    """Build nine numeric edge attributes for temporal or flattened graphs."""

    def build(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        feature_names: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        return build_combined_edge_features(node_features, edge_index, feature_names)

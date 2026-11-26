"""
ChickenBehaviorLab Graph Construction
=====================================

Build temporal skeleton graphs using the shared combined edge-feature contract.
"""

from __future__ import annotations

import numpy as np

from chicken_behavior_lab.features.temporal_features import TemporalFeatureSequence
from chicken_behavior_lab.graph.edge_features import EdgeFeatureExtractor
from chicken_behavior_lab.graph.graph import TemporalSkeletonGraph
from chicken_behavior_lab.skeleton.topology import canonicalize_edge_index


class SkeletonGraphBuilder:
    """Build temporal skeleton graphs from canonical integer connections."""

    def __init__(
        self,
        skeleton_connections: list[tuple[int, int]] | tuple[tuple[int, int], ...],
        directed: bool = False,
        edge_feature_extractor: EdgeFeatureExtractor | None = None,
    ) -> None:
        if not skeleton_connections:
            raise ValueError("skeleton_connections cannot be empty.")
        self.skeleton_connections = tuple(skeleton_connections)
        self.directed = directed
        self.edge_feature_extractor = edge_feature_extractor or EdgeFeatureExtractor()

    def build_edge_index(self) -> np.ndarray:
        """Convert canonical connections into validated directed edge_index."""
        raw_edges = np.asarray(self.skeleton_connections, dtype=np.int64).T
        return canonicalize_edge_index(
            raw_edges,
            bidirectional=not self.directed,
        )

    def build(self, sequence: TemporalFeatureSequence) -> TemporalSkeletonGraph:
        if not isinstance(sequence, TemporalFeatureSequence):
            raise TypeError("sequence must be a TemporalFeatureSequence.")
        edge_index = self.build_edge_index()
        if edge_index.size and np.any(edge_index >= sequence.num_keypoints):
            raise ValueError("Skeleton connection refers to a keypoint outside the feature sequence.")

        edge_features = self.edge_feature_extractor.extract(
            node_features=sequence.features,
            edge_index=edge_index,
            feature_names=sequence.feature_names,
        )
        edge_valid = (
            sequence.keypoint_mask[:, edge_index[0]]
            & sequence.keypoint_mask[:, edge_index[1]]
            & sequence.frame_mask[:, None]
        )
        edge_features[~edge_valid] = 0.0

        graph = TemporalSkeletonGraph(
            node_features=sequence.features.copy(),
            edge_index=edge_index,
            frame_mask=sequence.frame_mask.copy(),
            node_mask=sequence.keypoint_mask.copy(),
            edge_features=edge_features,
        )
        graph.validate()
        return graph

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
        edges: list[tuple[int, int]] = []
        for source, target in self.skeleton_connections:
            if source < 0 or target < 0:
                raise ValueError("Skeleton node indices must be non-negative.")
            if source == target:
                raise ValueError("Skeleton connections cannot contain self-loops.")
            edges.append((source, target))
            if not self.directed:
                edges.append((target, source))
        if len(set(edges)) != len(edges):
            raise ValueError("Skeleton connections contain duplicate directed edges.")
        return np.asarray(edges, dtype=np.int64).T

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

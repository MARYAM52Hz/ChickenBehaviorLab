from __future__ import annotations

import numpy as np

from chicken_behavior_lab.features.edge_contract import build_combined_edge_features
from chicken_behavior_lab.features.temporal_features import TemporalFeatureSequence
from chicken_behavior_lab.graphs.spatial_edges import SpatialEdgeExpander
from chicken_behavior_lab.graphs.spatiotemporal_graph import SpatioTemporalGraph
from chicken_behavior_lab.graphs.temporal_edges import TemporalEdgeBuilder


class SpatioTemporalGraphBuilder:
    """Build a flattened graph with spatial and temporal typed edges."""

    SPATIAL_EDGE = 0
    TEMPORAL_EDGE = 1

    def __init__(
        self,
        skeleton_edge_index: np.ndarray,
        bidirectional_temporal_edges: bool = True,
    ) -> None:
        self.skeleton_edge_index = np.asarray(skeleton_edge_index, dtype=np.int64)
        self.spatial_edge_expander = SpatialEdgeExpander()
        self.temporal_edge_builder = TemporalEdgeBuilder(
            bidirectional=bidirectional_temporal_edges
        )

    def build(self, sequence: TemporalFeatureSequence) -> SpatioTemporalGraph:
        if not isinstance(sequence, TemporalFeatureSequence):
            raise TypeError("sequence must be a TemporalFeatureSequence.")
        features = np.asarray(sequence.features, dtype=np.float32)
        if features.ndim != 3:
            raise ValueError("sequence.features must have shape (T, V, F).")
        num_frames, num_keypoints, feature_dim = features.shape
        node_features = features.reshape(num_frames * num_keypoints, feature_dim)

        spatial_edges = self.spatial_edge_expander.build(
            self.skeleton_edge_index, num_frames, num_keypoints
        )
        temporal_edges = self.temporal_edge_builder.build(num_frames, num_keypoints)
        edge_index = np.concatenate([spatial_edges, temporal_edges], axis=1)
        edge_type = np.concatenate([
            np.full(spatial_edges.shape[1], self.SPATIAL_EDGE, dtype=np.int64),
            np.full(temporal_edges.shape[1], self.TEMPORAL_EDGE, dtype=np.int64),
        ])

        node_mask = (sequence.keypoint_mask & sequence.frame_mask[:, None]).reshape(-1).copy()
        edge_features = build_combined_edge_features(
            node_features, edge_index, sequence.feature_names
        )
        if edge_index.shape[1]:
            edge_valid = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            edge_features[~edge_valid] = 0.0

        graph = SpatioTemporalGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            edge_type=edge_type,
            frame_ids=tuple(sequence.frame_ids),
            num_frames=num_frames,
            num_keypoints=num_keypoints,
            frame_mask=sequence.frame_mask.copy(),
            node_mask=node_mask,
        )
        graph.validate()
        return graph

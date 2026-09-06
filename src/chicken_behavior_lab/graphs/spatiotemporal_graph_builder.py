from __future__ import annotations

import numpy as np

from chicken_behavior_lab.features.temporal_features import (
    TemporalFeatureSequence,
)

from chicken_behavior_lab.graphs.spatial_edges import (
    SpatialEdgeExpander,
)

from chicken_behavior_lab.graphs.spatiotemporal_graph import (
    SpatioTemporalGraph,
)

from chicken_behavior_lab.graphs.temporal_edges import (
    TemporalEdgeBuilder,
)


class SpatioTemporalGraphBuilder:
    """
    Convert a TemporalFeatureSequence into one flattened
    spatio-temporal graph.

    The resulting graph contains:

    1. spatial skeleton edges within each frame
    2. temporal edges between matching keypoints
       in consecutive frames
    """

    SPATIAL_EDGE = 0
    TEMPORAL_EDGE = 1

    def __init__(
        self,
        skeleton_edge_index: np.ndarray,
        bidirectional_temporal_edges: bool = True,
    ) -> None:

        self.skeleton_edge_index = np.asarray(
            skeleton_edge_index,
            dtype=np.int64,
        )

        self.spatial_edge_expander = (
            SpatialEdgeExpander()
        )

        self.temporal_edge_builder = (
            TemporalEdgeBuilder(
                bidirectional=(
                    bidirectional_temporal_edges
                )
            )
        )

    def build(
        self,
        sequence: TemporalFeatureSequence,
    ) -> SpatioTemporalGraph:
        """
        Build one spatio-temporal skeleton graph.
        """

        if not isinstance(
            sequence,
            TemporalFeatureSequence,
        ):
            raise TypeError(
                "sequence must be a "
                "TemporalFeatureSequence."
            )

        features = np.asarray(
            sequence.features,
            dtype=np.float32,
        )

        if features.ndim != 3:
            raise ValueError(
                "sequence.features must have "
                "shape (T, V, F)."
            )

        num_frames = (
            features.shape[0]
        )

        num_keypoints = (
            features.shape[1]
        )

        feature_dim = (
            features.shape[2]
        )

        # =================================================
        # Flatten nodes
        # =================================================

        node_features = (
            features.reshape(
                num_frames
                * num_keypoints,
                feature_dim,
            )
        )

        # =================================================
        # Spatial edges
        # =================================================

        spatial_edges = (
            self.spatial_edge_expander.build(
                skeleton_edge_index=(
                    self.skeleton_edge_index
                ),
                num_frames=num_frames,
                num_keypoints=num_keypoints,
            )
        )

        # =================================================
        # Temporal edges
        # =================================================

        temporal_edges = (
            self.temporal_edge_builder.build(
                num_frames=num_frames,
                num_keypoints=num_keypoints,
            )
        )

        # =================================================
        # Combine edges
        # =================================================

        edge_index = np.concatenate(
            [
                spatial_edges,
                temporal_edges,
            ],
            axis=1,
        )

        # =================================================
        # Edge types
        # =================================================

        spatial_edge_type = np.full(
            spatial_edges.shape[1],
            self.SPATIAL_EDGE,
            dtype=np.int64,
        )

        temporal_edge_type = np.full(
            temporal_edges.shape[1],
            self.TEMPORAL_EDGE,
            dtype=np.int64,
        )

        edge_type = np.concatenate(
            [
                spatial_edge_type,
                temporal_edge_type,
            ]
        )

        # =================================================
        # Edge features
        # =================================================

        edge_features = (
            self._build_edge_features(
                node_features=node_features,
                edge_index=edge_index,
                edge_type=edge_type,
            )
        )

        # =================================================
        # Graph
        # =================================================

        graph = SpatioTemporalGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            edge_type=edge_type,
            frame_ids=tuple(
                sequence.frame_ids
            ),
            num_frames=num_frames,
            num_keypoints=num_keypoints,
        )

        graph.validate()

        return graph

    @staticmethod
    def _build_edge_features(
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_type: np.ndarray,
    ) -> np.ndarray:
        """
        Build initial edge attributes.

        Edge feature:

            target_features - source_features

        plus one additional edge-type feature.

        Therefore:

            D = F + 1

        Last column:

            0 = spatial
            1 = temporal
        """

        if edge_index.shape[1] == 0:

            feature_dim = (
                node_features.shape[1]
            )

            return np.empty(
                (
                    0,
                    feature_dim + 1,
                ),
                dtype=np.float32,
            )

        source = edge_index[0]
        target = edge_index[1]

        feature_difference = (
            node_features[target]
            - node_features[source]
        )

        edge_type_feature = (
            edge_type.astype(
                np.float32
            )
            .reshape(
                -1,
                1,
            )
        )

        return np.concatenate(
            [
                feature_difference,
                edge_type_feature,
            ],
            axis=1,
        )

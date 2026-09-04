from __future__ import annotations

import numpy as np

from chicken_behavior_lab.features.temporal_features import (
    TemporalFeatureSequence,
)

from chicken_behavior_lab.graphs.edge_features import (
    EdgeFeatureBuilder,
)

from chicken_behavior_lab.graphs.temporal_skeleton_graph import (
    TemporalSkeletonGraph,
)


class TemporalSkeletonGraphBuilder:
    """
    Convert a TemporalFeatureSequence into a
    TemporalSkeletonGraph.
    """

    def __init__(
        self,
        edge_index: np.ndarray,
    ) -> None:

        self.edge_index = np.asarray(
            edge_index,
            dtype=np.int64,
        )

        self.edge_feature_builder = (
            EdgeFeatureBuilder()
        )

    def build(
        self,
        sequence: TemporalFeatureSequence,
    ) -> TemporalSkeletonGraph:
        """
        Build a temporal skeleton graph.
        """

        if not isinstance(
            sequence,
            TemporalFeatureSequence,
        ):
            raise TypeError(
                "sequence must be a "
                "TemporalFeatureSequence."
            )

        node_features = (
            np.asarray(
                sequence.features,
                dtype=np.float32,
            )
        )

        edge_features = (
            self.edge_feature_builder.build(
                node_features,
                self.edge_index,
            )
        )

        graph = TemporalSkeletonGraph(
            node_features=node_features,
            edge_index=self.edge_index,
            edge_features=edge_features,
            frame_ids=sequence.frame_ids,
        )

        graph.validate()

        return graph

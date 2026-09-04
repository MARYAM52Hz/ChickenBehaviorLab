import numpy as np

from chicken_behavior_lab.features import (
    TemporalFeatureSequence,
)

from chicken_behavior_lab.graphs import (
    TemporalSkeletonGraphBuilder,
)


def test_temporal_graph_builder():

    features = np.array(
        [
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 3.0],
            ],
            [
                [0.5, 0.0],
                [1.5, 1.0],
                [2.5, 3.0],
            ],
        ],
        dtype=np.float32,
    )

    sequence = TemporalFeatureSequence(
        features=features,
        frame_mask=np.array(
            [True, True]
        ),
        keypoint_mask=np.ones(
            (2, 3),
            dtype=bool,
        ),
        frame_ids=(
            "100",
            "101",
        ),
    )

    edge_index = np.array(
        [
            [0, 1],
            [1, 2],
        ],
        dtype=np.int64,
    )

    builder = (
        TemporalSkeletonGraphBuilder(
            edge_index=edge_index
        )
    )

    graph = builder.build(
        sequence
    )

    assert graph.num_frames == 2
    assert graph.num_nodes == 3
    assert graph.num_edges == 2

    assert graph.edge_features.shape == (
        2,
        2,
        2,
    )

    np.testing.assert_allclose(
        graph.edge_features[0, 0],
        [1.0, 1.0],
    )

    np.testing.assert_allclose(
        graph.edge_features[0, 1],
        [1.0, 2.0],
    )

import numpy as np

from chicken_behavior_lab.features import (
    TemporalFeatureSequence,
)

from chicken_behavior_lab.graphs import (
    SpatioTemporalGraphBuilder,
)


def create_sequence():

    features = np.array(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            [
                [0.1, 0.0],
                [1.1, 0.0],
                [2.1, 0.0],
            ],
        ],
        dtype=np.float32,
    )

    return TemporalFeatureSequence(
        features=features,
        frame_mask=np.array(
            [
                True,
                True,
            ]
        ),
        keypoint_mask=np.ones(
            (
                2,
                3,
            ),
            dtype=bool,
        ),
        frame_ids=(
            "100",
            "101",
        ),
    )


def test_spatiotemporal_graph():

    sequence = create_sequence()

    skeleton_edges = np.array(
        [
            [0, 1],
            [1, 2],
        ],
        dtype=np.int64,
    )

    builder = (
        SpatioTemporalGraphBuilder(
            skeleton_edge_index=(
                skeleton_edges
            ),
            bidirectional_temporal_edges=True,
        )
    )

    graph = builder.build(
        sequence
    )

    # ----------------------------------------------
    # Nodes
    # ----------------------------------------------

    assert graph.num_frames == 2

    assert graph.num_keypoints == 3

    assert graph.num_nodes == 6

    assert graph.node_features.shape == (
        6,
        2,
    )

    # ----------------------------------------------
    # Spatial edges
    # ----------------------------------------------

    # 2 spatial edges per frame
    #
    # 2 frames × 2 edges
    #
    # = 4

    assert (
        graph.num_spatial_edges
        == 4
    )

    # ----------------------------------------------
    # Temporal edges
    # ----------------------------------------------

    # 3 keypoints
    #
    # bidirectional:
    #
    # 3 × 2
    #
    # = 6

    assert (
        graph.num_temporal_edges
        == 6
    )

    # ----------------------------------------------
    # Total
    # ----------------------------------------------

    assert graph.num_edges == 10

    # ----------------------------------------------
    # Edge features
    # ----------------------------------------------

    # F = 2
    #
    # + edge type = 1
    #
    # therefore D = 3

    assert graph.edge_features.shape == (
        10,
        3,
    )


def test_temporal_edge_type():

    sequence = create_sequence()

    skeleton_edges = np.array(
        [
            [0],
            [1],
        ],
        dtype=np.int64,
    )

    builder = (
        SpatioTemporalGraphBuilder(
            skeleton_edge_index=(
                skeleton_edges
            )
        )
    )

    graph = builder.build(
        sequence
    )

    assert set(
        np.unique(
            graph.edge_type
        ).tolist()
    ) == {
        0,
        1,
    }


def test_frame_ids_preserved():

    sequence = create_sequence()

    skeleton_edges = np.array(
        [
            [0],
            [1],
        ],
        dtype=np.int64,
    )

    builder = (
        SpatioTemporalGraphBuilder(
            skeleton_edge_index=(
                skeleton_edges
            )
        )
    )

    graph = builder.build(
        sequence
    )

    assert graph.frame_ids == (
        "100",
        "101",
    )

"""
Tests for ChickenBehaviorLab graph construction.
"""

import numpy as np

from chicken_behavior_lab.features.temporal_features import (
    TemporalFeatureSequence,
)

from chicken_behavior_lab.graph.graph import (
    SkeletonGraph,
    TemporalSkeletonGraph,
)

from chicken_behavior_lab.graph.construction import (
    SkeletonGraphBuilder,
)


def create_sequence() -> TemporalFeatureSequence:

    features = np.zeros(
        (
            4,
            3,
            7,
        ),
        dtype=np.float32,
    )

    frame_mask = np.ones(
        4,
        dtype=bool,
    )

    keypoint_mask = np.ones(
        (
            4,
            3,
        ),
        dtype=bool,
    )

    return TemporalFeatureSequence(
        features=features,
        frame_mask=frame_mask,
        keypoint_mask=keypoint_mask,
        frame_ids=(
            "0",
            "1",
            "2",
            "3",
        ),
    )


def test_edge_index_is_undirected():

    builder = SkeletonGraphBuilder(
        skeleton_connections=[
            (0, 1),
            (1, 2),
        ]
    )

    edge_index = (
        builder.build_edge_index()
    )

    expected = np.array(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=np.int64,
    )

    assert np.array_equal(
        edge_index,
        expected,
    )


def test_temporal_graph_shape():

    sequence = create_sequence()

    builder = SkeletonGraphBuilder(
        skeleton_connections=[
            (0, 1),
            (1, 2),
        ]
    )

    graph = builder.build(
        sequence
    )

    assert (
        graph.node_features.shape
        == (4, 3, 7)
    )

    assert (
        graph.edge_index.shape
        == (2, 4)
    )


def test_temporal_graph_masks():

    sequence = create_sequence()

    sequence.frame_mask[
        0
    ] = False

    sequence.keypoint_mask[
        1,
        2,
    ] = False

    builder = SkeletonGraphBuilder(
        skeleton_connections=[
            (0, 1),
            (1, 2),
        ]
    )

    graph = builder.build(
        sequence
    )

    assert (
        graph.frame_mask[0]
        is False
    )

    assert (
        graph.node_mask[1, 2]
        is False
    )


def test_graph_validation():

    graph = SkeletonGraph(
        node_features=np.zeros(
            (3, 7),
            dtype=np.float32,
        ),
        edge_index=np.array(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=np.int64,
        ),
    )

    graph.validate()

    assert graph.num_nodes == 3
    assert graph.num_edges == 2

"""
Tests for ChickenBehaviorLab graph dataset.
"""

import numpy as np

from chicken_behavior_lab.dataset import (
    GraphDataset,
    GraphSample,
)

from chicken_behavior_lab.graph import (
    SkeletonGraphBuilder,
)

from chicken_behavior_lab.features.temporal_features import (
    TemporalFeatureSequence,
)


def create_graph():

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

    sequence = TemporalFeatureSequence(
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

    builder = SkeletonGraphBuilder(
        skeleton_connections=[
            (0, 1),
            (1, 2),
        ]
    )

    return builder.build(
        sequence
    )


def test_graph_sample():

    graph = create_graph()

    sample = GraphSample(
        graph=graph,
        label=2,
        sample_id="sample_001",
    )

    sample.validate()

    assert sample.label == 2

    assert (
        sample.sample_id
        == "sample_001"
    )


def test_graph_dataset():

    graph_1 = create_graph()
    graph_2 = create_graph()

    sample_1 = GraphSample(
        graph=graph_1,
        label=0,
        sample_id="sample_001",
    )

    sample_2 = GraphSample(
        graph=graph_2,
        label=1,
        sample_id="sample_002",
    )

    dataset = GraphDataset(
        [
            sample_1,
            sample_2,
        ]
    )

    assert len(dataset) == 2

    assert dataset.labels == [
        0,
        1,
    ]

    assert dataset.sample_ids == [
        "sample_001",
        "sample_002",
    ]


def test_dataset_indexing():

    graph = create_graph()

    sample = GraphSample(
        graph=graph,
        label=3,
        sample_id="sample_001",
    )

    dataset = GraphDataset(
        [sample]
    )

    retrieved = dataset[0]

    assert (
        retrieved.sample_id
        == "sample_001"
    )

    assert (
        retrieved.label
        == 3
    )


def test_duplicate_sample_ids():

    graph = create_graph()

    sample_1 = GraphSample(
        graph=graph,
        label=0,
        sample_id="duplicate",
    )

    sample_2 = GraphSample(
        graph=graph,
        label=1,
        sample_id="duplicate",
    )

    try:

        GraphDataset(
            [
                sample_1,
                sample_2,
            ]
        )

    except ValueError as error:

        assert (
            "Duplicate sample_id"
            in str(error)
        )

    else:

        raise AssertionError(
            "Expected duplicate "
            "sample_id error."
        )

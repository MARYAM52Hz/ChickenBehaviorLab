"""
Tests for ChickenBehaviorLab graph dataset.

This module verifies:

1. GraphSample creation
2. GraphSample validation
3. GraphDataset creation
4. Dataset indexing
5. Dataset iteration
6. Label handling
7. Sample ID handling
8. Duplicate sample ID detection
"""

from __future__ import annotations

import numpy as np
import pytest

from chicken_behavior_lab.dataset import (
    GraphDataset,
    GraphSample,
)

from chicken_behavior_lab.features.temporal_features import (
    TemporalFeatureSequence,
)

from chicken_behavior_lab.graph import (
    SkeletonGraphBuilder,
)


# =========================================================
# Test Helpers
# =========================================================

def create_graph():
    """
    Create a small deterministic temporal skeleton graph
    for testing purposes.

    Returns
    -------
    TemporalSkeletonGraph
        A graph with:

            T = 4 frames
            V = 3 keypoints
            F = 7 node features

        Skeleton connections:

            0 <-> 1
            1 <-> 2
    """

    # -----------------------------------------------------
    # Temporal node features
    # -----------------------------------------------------

    features = np.zeros(
        (
            4,  # temporal length
            3,  # number of keypoints
            7,  # feature dimension
        ),
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # Frame validity mask
    # -----------------------------------------------------

    frame_mask = np.ones(
        4,
        dtype=bool,
    )

    # -----------------------------------------------------
    # Keypoint validity mask
    # -----------------------------------------------------

    keypoint_mask = np.ones(
        (
            4,
            3,
        ),
        dtype=bool,
    )

    # -----------------------------------------------------
    # Frame IDs
    # -----------------------------------------------------

    frame_ids = (
        "0",
        "1",
        "2",
        "3",
    )

    # -----------------------------------------------------
    # Create temporal feature sequence
    # -----------------------------------------------------

    sequence = TemporalFeatureSequence(
        features=features,
        frame_mask=frame_mask,
        keypoint_mask=keypoint_mask,
        frame_ids=frame_ids,
    )

    # -----------------------------------------------------
    # Create graph builder
    # -----------------------------------------------------

    builder = SkeletonGraphBuilder(
        skeleton_connections=[
            (0, 1),
            (1, 2),
        ]
    )

    # -----------------------------------------------------
    # Build graph
    # -----------------------------------------------------

    return builder.build(
        sequence
    )


# =========================================================
# GraphSample Tests
# =========================================================

def test_graph_sample_creation():
    """
    Verify that a GraphSample can be created correctly.
    """

    graph = create_graph()

    sample = GraphSample(
        graph=graph,
        label=0,
        behavior_id="feeding",
        sample_id="sample_001",
    )

    assert sample.graph is graph

    assert sample.label == 0

    assert (
        sample.behavior_id
        == "feeding"
    )

    assert (
        sample.sample_id
        == "sample_001"
    )


# =========================================================
# GraphSample Validation
# =========================================================

def test_graph_sample_validation():
    """
    Verify GraphSample validation.
    """

    graph = create_graph()

    sample = GraphSample(
        graph=graph,
        label=1,
        behavior_id="walking",
        sample_id="sample_001",
    )

    sample.validate()

    assert (
        sample.label
        == 1
    )

    assert (
        sample.behavior_id
        == "walking"
    )


# =========================================================
# GraphSample Negative Label
# =========================================================

def test_negative_label_is_rejected():
    """
    Verify that negative class indices are rejected.
    """

    graph = create_graph()

    sample = GraphSample(
        graph=graph,
        label=-1,
        behavior_id="feeding",
        sample_id="sample_001",
    )

    with pytest.raises(
        ValueError
    ):

        sample.validate()


# =========================================================
# GraphSample Invalid Label Type
# =========================================================

def test_invalid_label_type_is_rejected():
    """
    Verify that non-integer labels are rejected.
    """

    graph = create_graph()

    sample = GraphSample(
        graph=graph,
        label="feeding",
        behavior_id="feeding",
        sample_id="sample_001",
    )

    with pytest.raises(
        TypeError
    ):

        sample.validate()


# =========================================================
# Empty Behavior ID
# =========================================================

def test_empty_behavior_id_is_rejected():
    """
    Verify that behavior_id cannot be empty.
    """

    graph = create_graph()

    sample = GraphSample(
        graph=graph,
        label=0,
        behavior_id="",
        sample_id="sample_001",
    )

    with pytest.raises(
        ValueError
    ):

        sample.validate()


# =========================================================
# Empty Sample ID
# =========================================================

def test_empty_sample_id_is_rejected():
    """
    Verify that sample_id cannot be empty.
    """

    graph = create_graph()

    sample = GraphSample(
        graph=graph,
        label=0,
        behavior_id="feeding",
        sample_id="",
    )

    with pytest.raises(
        ValueError
    ):

        sample.validate()


# =========================================================
# GraphDataset Creation
# =========================================================

def test_graph_dataset_creation():
    """
    Verify GraphDataset creation with multiple samples.
    """

    graph_1 = create_graph()

    graph_2 = create_graph()

    sample_1 = GraphSample(
        graph=graph_1,
        label=0,
        behavior_id="feeding",
        sample_id="sample_001",
    )

    sample_2 = GraphSample(
        graph=graph_2,
        label=1,
        behavior_id="walking",
        sample_id="sample_002",
    )

    dataset = GraphDataset(
        [
            sample_1,
            sample_2,
        ]
    )

    assert len(dataset) == 2


# =========================================================
# Dataset Labels
# =========================================================

def test_dataset_labels():
    """
    Verify that dataset.labels returns the correct
    integer class indices.
    """

    graph_1 = create_graph()

    graph_2 = create_graph()

    graph_3 = create_graph()

    sample_1 = GraphSample(
        graph=graph_1,
        label=0,
        behavior_id="feeding",
        sample_id="sample_001",
    )

    sample_2 = GraphSample(
        graph=graph_2,
        label=1,
        behavior_id="walking",
        sample_id="sample_002",
    )

    sample_3 = GraphSample(
        graph=graph_3,
        label=2,
        behavior_id="standing",
        sample_id="sample_003",
    )

    dataset = GraphDataset(
        [
            sample_1,
            sample_2,
            sample_3,
        ]
    )

    assert dataset.labels == [
        0,
        1,
        2,
    ]


# =========================================================
# Dataset Sample IDs
# =========================================================

def test_dataset_sample_ids():
    """
    Verify that dataset.sample_ids returns all IDs
    in dataset order.
    """

    graph_1 = create_graph()

    graph_2 = create_graph()

    sample_1 = GraphSample(
        graph=graph_1,
        label=0,
        behavior_id="feeding",
        sample_id="sample_001",
    )

    sample_2 = GraphSample(
        graph=graph_2,
        label=1,
        behavior_id="walking",
        sample_id="sample_002",
    )

    dataset = GraphDataset(
        [
            sample_1,
            sample_2,
        ]
    )

    assert dataset.sample_ids == [
        "sample_001",
        "sample_002",
    ]


# =========================================================
# Dataset Indexing
# =========================================================

def test_dataset_indexing():
    """
    Verify dataset indexing.
    """

    graph = create_graph()

    sample = GraphSample(
        graph=graph,
        label=3,
        behavior_id="drinking",
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

    assert (
        retrieved.behavior_id
        == "drinking"
    )


# =========================================================
# Dataset Iteration
# =========================================================

def test_dataset_iteration():
    """
    Verify iteration over GraphDataset.
    """

    graph_1 = create_graph()

    graph_2 = create_graph()

    sample_1 = GraphSample(
        graph=graph_1,
        label=0,
        behavior_id="feeding",
        sample_id="sample_001",
    )

    sample_2 = GraphSample(
        graph=graph_2,
        label=1,
        behavior_id="walking",
        sample_id="sample_002",
    )

    dataset = GraphDataset(
        [
            sample_1,
            sample_2,
        ]
    )

    retrieved_ids = [
        sample.sample_id
        for sample in dataset
    ]

    assert retrieved_ids == [
        "sample_001",
        "sample_002",
    ]


# =========================================================
# Duplicate Sample IDs
# =========================================================

def test_duplicate_sample_ids():
    """
    Verify that duplicate sample IDs are rejected.
    """

    graph_1 = create_graph()

    graph_2 = create_graph()

    sample_1 = GraphSample(
        graph=graph_1,
        label=0,
        behavior_id="feeding",
        sample_id="duplicate",
    )

    sample_2 = GraphSample(
        graph=graph_2,
        label=1,
        behavior_id="walking",
        sample_id="duplicate",
    )

    with pytest.raises(
        ValueError,
        match="Duplicate sample_id",
    ):

        GraphDataset(
            [
                sample_1,
                sample_2,
            ]
        )


# =========================================================
# Empty Dataset
# =========================================================

def test_empty_dataset():
    """
    Verify that an empty GraphDataset can be created.
    """

    dataset = GraphDataset(
        []
    )

    assert len(dataset) == 0

    assert dataset.labels == []

    assert dataset.sample_ids == []


# =========================================================
# Graph Integrity Inside Dataset
# =========================================================

def test_graph_integrity_inside_dataset():
    """
    Verify that the graph stored inside a sample
    retains its expected dimensions.
    """

    graph = create_graph()

    sample = GraphSample(
        graph=graph,
        label=0,
        behavior_id="feeding",
        sample_id="sample_001",
    )

    dataset = GraphDataset(
        [sample]
    )

    retrieved = dataset[0]

    assert (
        retrieved.graph.temporal_length
        == 4
    )

    assert (
        retrieved.graph.num_nodes
        == 3
    )

    assert (
        retrieved.graph.num_edges
        == 4
    )

    assert (
        retrieved.graph.edge_feature_dimension
        == 3
    )


# =========================================================
# Edge Feature Integrity
# =========================================================

def test_edge_features_inside_dataset():
    """
    Verify that edge features remain attached to the
    temporal graph after insertion into the dataset.
    """

    graph = create_graph()

    sample = GraphSample(
        graph=graph,
        label=0,
        behavior_id="feeding",
        sample_id="sample_001",
    )

    dataset = GraphDataset(
        [sample]
    )

    retrieved = dataset[0]

    assert (
        retrieved.graph.edge_features
        is not None
    )

    assert (
        retrieved.graph.edge_features.shape
        == (4, 4, 3)
    )


# =========================================================
# Dataset Ordering
# =========================================================

def test_dataset_order_is_preserved():
    """
    Verify that GraphDataset preserves insertion order.
    """

    graph_1 = create_graph()

    graph_2 = create_graph()

    graph_3 = create_graph()

    samples = [
        GraphSample(
            graph=graph_1,
            label=2,
            behavior_id="standing",
            sample_id="sample_003",
        ),
        GraphSample(
            graph=graph_2,
            label=0,
            behavior_id="feeding",
            sample_id="sample_001",
        ),
        GraphSample(
            graph=graph_3,
            label=1,
            behavior_id="walking",
            sample_id="sample_002",
        ),
    ]

    dataset = GraphDataset(
        samples
    )

    assert dataset.sample_ids == [
        "sample_003",
        "sample_001",
        "sample_002",
    ]

    assert dataset.labels == [
        2,
        0,
        1,
    ]

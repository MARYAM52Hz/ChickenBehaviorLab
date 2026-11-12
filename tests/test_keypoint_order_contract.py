from __future__ import annotations

import numpy as np
import pytest

from chicken_behavior_lab.core.enums.keypoints import KeypointType
from chicken_behavior_lab.features.motion import MotionFeatureFrame
from chicken_behavior_lab.features.temporal_features import TemporalFeatureBuilder
from chicken_behavior_lab.graphs.temporal_graph_builder import TemporalSkeletonGraphBuilder
from chicken_behavior_lab.graphs.temporal_skeleton_graph import TemporalSkeletonGraph
from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.dataset.temporal_pyg_dataset import TemporalPyGDataset


def make_frame(order: tuple[KeypointType, ...]) -> MotionFeatureFrame:
    return MotionFeatureFrame(
        features=np.zeros((len(order), 7), dtype=np.float32),
        angles=np.empty((0,), dtype=np.float32),
        angle_valid_mask=np.empty((0,), dtype=bool),
        feature_names=("x", "y", "vx", "vy", "ax", "ay", "confidence"),
        keypoint_mask=np.ones((len(order),), dtype=bool),
        keypoint_types=order,
    )


def test_temporal_feature_builder_preserves_keypoint_order():
    order = (KeypointType.BEAK, KeypointType.HEAD)
    sequence = TemporalFeatureBuilder().build(
        [make_frame(order), make_frame(order)],
        frame_ids=["0", "1"],
    )

    assert sequence.keypoint_types == order

    graph = TemporalSkeletonGraphBuilder(
        edge_index=np.asarray([[0], [1]], dtype=np.int64)
    ).build(sequence)

    assert graph.keypoint_types == order


def test_temporal_feature_builder_rejects_order_changes_between_frames():
    first = (KeypointType.BEAK, KeypointType.HEAD)
    second = (KeypointType.HEAD, KeypointType.BEAK)

    with pytest.raises(ValueError, match="Inconsistent keypoint order"):
        TemporalFeatureBuilder().build(
            [make_frame(first), make_frame(second)]
        )


def test_temporal_graph_rejects_keypoint_order_with_wrong_node_count():
    order = (KeypointType.BEAK, KeypointType.HEAD)
    sequence = TemporalFeatureBuilder().build([make_frame(order)])
    graph = TemporalSkeletonGraphBuilder(
        edge_index=np.asarray([[0], [1]], dtype=np.int64)
    ).build(sequence)

    graph.keypoint_types = (KeypointType.BEAK,)
    with pytest.raises(ValueError, match="length must match"):
        graph.validate()


def make_graph(order: tuple[KeypointType, ...]) -> TemporalSkeletonGraph:
    return TemporalSkeletonGraph(
        node_features=np.zeros((1, len(order), 1), dtype=np.float32),
        edge_index=np.asarray([[0], [1]], dtype=np.int64),
        edge_features=np.zeros((1, 1, 1), dtype=np.float32),
        frame_ids=("0",),
        frame_mask=np.ones((1,), dtype=bool),
        node_mask=np.ones((1, len(order)), dtype=bool),
        keypoint_types=order,
    )


def make_graph_sample(sample_id: str, order: tuple[KeypointType, ...]) -> GraphSample:
    return GraphSample(
        graph=make_graph(order),
        label=0,
        behavior_id="feeding",
        sample_id=sample_id,
        metadata={
            "video_id": "video_001",
            "track_id": "track_1",
            "start_frame": 0,
            "end_frame": 0,
        },
    )


def test_temporal_dataset_rejects_different_keypoint_orders():
    first = (KeypointType.BEAK, KeypointType.HEAD)
    second = (KeypointType.HEAD, KeypointType.BEAK)

    with pytest.raises(ValueError, match="same keypoint order"):
        TemporalPyGDataset(
            [
                make_graph_sample("sample_1", first),
                make_graph_sample("sample_2", second),
            ]
        )

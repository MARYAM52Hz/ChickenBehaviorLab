from __future__ import annotations

import numpy as np
import pytest

from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.dataset.temporal_pyg_dataset import TemporalPyGDataset
from chicken_behavior_lab.graph.graph import TemporalSkeletonGraph


def make_sample(sample_id="sample_01", length=5, behavior_id="feeding", label=0):
    nodes = np.arange(length * 3 * 4, dtype=np.float32).reshape(length, 3, 4)
    frame_mask = np.ones(length, dtype=bool)
    node_mask = np.ones((length, 3), dtype=bool)
    node_mask[1, 2] = False
    graph = TemporalSkeletonGraph(
        node_features=nodes,
        edge_index=np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
        frame_mask=frame_mask,
        node_mask=node_mask,
        edge_features=np.ones((length, 4, 2), dtype=np.float32),
    )
    return GraphSample(
        graph=graph,
        label=label,
        behavior_id=behavior_id,
        sample_id=sample_id,
        metadata={
            "video_id": "video_001",
            "track_id": 1,
            "start_frame": 100,
            "end_frame": 100 + length - 1,
        },
    )


def test_dataset_keeps_one_annotation_as_one_temporal_sample():
    dataset = TemporalPyGDataset([make_sample()])
    item = dataset[0]
    assert len(dataset) == 1
    assert item.x.shape == (5, 3, 4)
    assert item.edge_attr.shape == (5, 4, 2)
    assert item.frame_mask.shape == (5,)
    assert item.node_mask.shape == (5, 3)
    assert item.edge_mask.shape == (5, 4)
    assert item.track_id == "track_1"
    assert item.start_frame == 100
    assert item.end_frame == 104
    assert item.node_mask[1].tolist() == [True, True, False]
    assert item.edge_mask[1].tolist() == [True, True, False, False]


def test_validation_dataset_uses_training_mapping():
    train = TemporalPyGDataset([make_sample()])
    validation = TemporalPyGDataset(
        [make_sample(sample_id="sample_02")],
        label_to_index=train.label_to_index,
    )
    assert validation.label_to_index == {"feeding": 0}


def test_dataset_rejects_inconsistent_external_mapping():
    with pytest.raises(ValueError, match="Inconsistent label mapping"):
        TemporalPyGDataset(
            [make_sample()],
            label_to_index={"feeding": 5},
        )


def test_dataset_rejects_temporal_length_not_matching_inclusive_interval():
    sample = make_sample(length=5)
    sample.metadata["end_frame"] = 106
    with pytest.raises(ValueError, match="inclusive frames"):
        TemporalPyGDataset([sample])[0]


def test_dataset_rejects_sample_without_valid_frames():
    sample = make_sample()
    sample.graph.frame_mask[:] = False
    with pytest.raises(ValueError, match="no valid frames"):
        TemporalPyGDataset([sample])[0]

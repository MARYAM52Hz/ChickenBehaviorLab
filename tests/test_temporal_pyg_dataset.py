from __future__ import annotations

import numpy as np
import pytest

from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.dataset.temporal_builder import TemporalSequenceBuilder
from chicken_behavior_lab.dataset.temporal_pyg_dataset import TemporalPyGDataset
from chicken_behavior_lab.graph.graph import SkeletonGraph


def make_sample(frame: int, *, behavior_id: str = "feeding", label: int = 0) -> GraphSample:
    graph = SkeletonGraph(
        node_features=np.full((13, 4), float(frame), dtype=np.float32),
        edge_index=np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        edge_features=np.full((3, 2), 0.1, dtype=np.float32),
        node_mask=np.ones((13,), dtype=bool),
    )
    return GraphSample(
        graph=graph,
        label=label,
        behavior_id=behavior_id,
        sample_id=f"video_001_track_1_frame_{frame:04d}",
        metadata={
            "video_id": "video_001",
            "track_id": 1,
            "start_frame": frame,
            "end_frame": frame,
        },
    )


def test_validation_dataset_uses_training_mapping() -> None:
    samples = [make_sample(frame) for frame in range(4)]
    temporal_dataset = TemporalSequenceBuilder(
        sequence_length=2,
        sequence_stride=2,
    ).build(samples)

    train_dataset = TemporalPyGDataset(temporal_dataset.samples)
    validation_dataset = TemporalPyGDataset(
        temporal_dataset.samples,
        label_to_index=train_dataset.label_to_index,
    )

    assert validation_dataset.label_to_index == train_dataset.label_to_index
    assert validation_dataset.label_to_index == {"feeding": 0}
    item = validation_dataset[0]
    assert item.x.shape == (2, 13, 4)
    assert item.edge_attr.shape == (2, 3, 2)
    assert item.track_id == "track_1"


def test_dataset_rejects_inconsistent_external_mapping() -> None:
    samples = [make_sample(frame) for frame in range(4)]
    temporal_dataset = TemporalSequenceBuilder(
        sequence_length=4,
    ).build(samples)

    with pytest.raises(ValueError, match="Inconsistent label mapping"):
        TemporalPyGDataset(
            temporal_dataset.samples,
            label_to_index={"feeding": 5},
        )

from __future__ import annotations

import numpy as np

from chicken_behavior_lab.dataset.group_splitter import split_dataset_by_group
from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.dataset.temporal_split_builder import build_temporal_splits
from chicken_behavior_lab.graph.graph import TemporalSkeletonGraph


class Dataset:
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def make_dataset():
    samples = []
    for video_number in range(1, 7):
        video_id = f"video_{video_number:03d}"
        for frame in range(40):
            graph = TemporalSkeletonGraph(
                node_features=np.asarray([[[float(frame), 0.0]]], dtype=np.float32),
                edge_index=np.empty((2, 0), dtype=np.int64),
            )
            samples.append(
                GraphSample(
                    graph=graph,
                    label=0 if frame < 20 else 1,
                    behavior_id="feeding" if frame < 20 else "walking",
                    sample_id=f"{video_id}_track_1_frame_{frame:04d}",
                    metadata={
                        "video_id": video_id,
                        "track_id": 1,
                        "start_frame": frame,
                        "end_frame": frame,
                    },
                )
            )
    return Dataset(samples)


def make_splits():
    dataset = make_dataset()
    return split_dataset_by_group(
        dataset,
        group_by="video",
        train_ratio=0.50,
        validation_ratio=0.25,
        test_ratio=0.25,
        seed=42,
    )


def test_temporal_windows_do_not_leak():
    temporal_splits = build_temporal_splits(
        make_splits(), sequence_length=8, sequence_stride=4,
    )
    train_groups = {(window.video_id, window.track_id) for window in temporal_splits.train}
    validation_groups = {(window.video_id, window.track_id) for window in temporal_splits.validation}
    test_groups = {(window.video_id, window.track_id) for window in temporal_splits.test}

    assert not (train_groups & validation_groups)
    assert not (train_groups & test_groups)
    assert not (validation_groups & test_groups)


def test_windows_are_temporal():
    temporal_splits = build_temporal_splits(
        make_splits(), sequence_length=8, sequence_stride=4,
    )
    for window in temporal_splits.train:
        assert len(window.samples) == 8
        assert window.end_frame >= window.start_frame
        assert window.end_frame - window.start_frame == 7


def test_homogeneous_behavior_windows_are_supported():
    temporal_splits = build_temporal_splits(
        make_splits(),
        sequence_length=8,
        sequence_stride=4,
        require_same_behavior=True,
    )
    assert temporal_splits.train
    assert all(len({sample.behavior_id for sample in window.samples}) == 1 for window in temporal_splits.train)

def test_three_way_dataset_split_stratifies_before_temporal_windowing():
    samples = []
    for video_number in range(9):
        video_id = f"behavior_video_{video_number:03d}"
        behaviors = []
        if video_number < 6:
            behaviors.append(("feeding", 0))
        if video_number >= 3:
            behaviors.append(("walking", 1))
        for behavior_id, label in behaviors:
            for frame in range(4):
                graph = TemporalSkeletonGraph(
                    node_features=np.asarray(
                        [[[float(frame), 0.0]]], dtype=np.float32
                    ),
                    edge_index=np.empty((2, 0), dtype=np.int64),
                )
                samples.append(
                    GraphSample(
                        graph=graph,
                        label=label,
                        behavior_id=behavior_id,
                        sample_id=f"{video_id}_{behavior_id}_{frame}",
                        metadata={
                            "video_id": video_id,
                            "track_id": 1,
                            "start_frame": frame,
                            "end_frame": frame,
                        },
                    )
                )

    dataset = Dataset(samples)
    splits = split_dataset_by_group(
        dataset,
        group_by="video",
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
        seed=31,
    )

    for subset in (splits.train, splits.validation, splits.test):
        labels = {subset[index].behavior_id for index in range(len(subset))}
        assert labels == {"feeding", "walking"}

    assert set(splits.train.indices).isdisjoint(splits.validation.indices)
    assert set(splits.train.indices).isdisjoint(splits.test.indices)
    assert set(splits.validation.indices).isdisjoint(splits.test.indices)


from __future__ import annotations

import numpy as np
import torch

from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.dataset.temporal_builder import TemporalSequenceBuilder
from chicken_behavior_lab.dataset.temporal_collate import make_temporal_dataloader
from chicken_behavior_lab.dataset.temporal_pyg_dataset import TemporalPyGDataset


class FrameGraph:
    def __init__(self, frame_id: int) -> None:
        rng = np.random.default_rng(frame_id)
        self.node_features = rng.normal(size=(13, 8)).astype(np.float32)
        self.edge_index = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        self.edge_features = rng.normal(size=(3, 4)).astype(np.float32)

    def validate(self) -> None:
        if self.node_features.shape != (13, 8) or self.edge_features.shape != (3, 4):
            raise ValueError("Unexpected graph feature shape.")


def make_samples(number_of_frames: int = 16) -> list[GraphSample]:
    return [
        GraphSample(
            graph=FrameGraph(index),
            label=0,
            behavior_id="feeding",
            sample_id=f"sample_{index}",
            metadata={
                "video_id": "video_001",
                "track_id": "track_1",
                "start_frame": index,
                "end_frame": index,
            },
        )
        for index in range(number_of_frames)
    ]


def make_loader(batch_size: int = 2):
    windows = TemporalSequenceBuilder(
        sequence_length=8,
        sequence_stride=4,
    ).build(make_samples())
    dataset = TemporalPyGDataset(windows.samples)
    return make_temporal_dataloader(dataset, batch_size=batch_size, shuffle=False)


def test_temporal_dataloader_shapes_and_track_identity():
    batch = next(iter(make_loader()))
    assert batch.x.shape == (2, 8, 13, 8)
    assert batch.edge_index.shape == (2, 3)
    assert batch.edge_attr.shape == (2, 8, 3, 4)
    assert batch.y.shape == (2,)
    assert len(batch.frame_batches) == 8
    assert batch.track_id == ["track_1", "track_1"]
    assert batch.frame_batches[0].num_graphs == 2
    assert batch.frame_batches[0].x.shape == (26, 8)
    assert batch.frame_batches[0].edge_attr.shape == (6, 4)


def test_temporal_batch_to_device_keeps_track_ids_as_strings():
    batch = next(iter(make_loader(batch_size=1)))
    moved = batch.to(torch.device("cpu"))
    assert moved.x.device.type == "cpu"
    assert moved.edge_index.device.type == "cpu"
    assert moved.frame_batches[0].x.device.type == "cpu"
    assert moved.track_id == ["track_1"]

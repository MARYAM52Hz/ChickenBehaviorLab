from __future__ import annotations

import numpy as np
import torch

from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.dataset.temporal_collate import make_temporal_dataloader
from chicken_behavior_lab.dataset.temporal_pyg_dataset import TemporalPyGDataset
from chicken_behavior_lab.graph.graph import TemporalSkeletonGraph


def make_sample(sample_id: str, length: int, start: int) -> GraphSample:
    rng = np.random.default_rng(start)
    graph = TemporalSkeletonGraph(
        node_features=rng.normal(size=(length, 13, 8)).astype(np.float32),
        edge_index=np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        frame_mask=np.ones(length, dtype=bool),
        node_mask=np.ones((length, 13), dtype=bool),
        edge_features=rng.normal(size=(length, 3, 4)).astype(np.float32),
    )
    return GraphSample(
        graph=graph,
        label=0,
        behavior_id="feeding",
        sample_id=sample_id,
        metadata={
            "video_id": f"video_{sample_id}",
            "track_id": "track_1",
            "start_frame": start,
            "end_frame": start + length - 1,
        },
    )


def make_loader(batch_size: int = 2):
    dataset = TemporalPyGDataset([
        make_sample("short", 5, 0),
        make_sample("long", 8, 10),
    ])
    return make_temporal_dataloader(dataset, batch_size=batch_size, shuffle=False)


def test_temporal_dataloader_shapes_and_track_identity():
    batch = next(iter(make_loader()))
    assert batch.x.shape == (2, 8, 13, 8)
    assert batch.edge_index.shape == (2, 3)
    assert batch.edge_attr.shape == (2, 8, 3, 4)
    assert batch.y.shape == (2,)
    assert len(batch.frame_batches) == 8
    assert batch.frame_mask[0].tolist() == [True] * 5 + [False] * 3
    assert batch.track_id == ["track_1", "track_1"]
    assert batch.frame_batches[0].num_graphs == 2
    assert batch.frame_batches[0].x.shape == (26, 8)
    assert batch.frame_batches[0].edge_attr.shape == (6, 4)


def test_temporal_batch_to_device_keeps_track_ids_as_strings():
    batch = next(iter(make_loader(batch_size=1)))
    moved = batch.to(torch.device("cpu"))
    assert moved.x.device.type == "cpu"
    assert moved.edge_index.device.type == "cpu"
    assert moved.frame_mask.device.type == "cpu"
    assert moved.frame_batches[0].x.device.type == "cpu"
    assert moved.track_id == ["track_1"]

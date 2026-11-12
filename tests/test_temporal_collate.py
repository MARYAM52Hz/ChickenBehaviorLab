from __future__ import annotations

import numpy as np
import torch

from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.dataset.temporal_pyg_dataset import TemporalPyGDataset
from chicken_behavior_lab.dataset.temporal_collate import TemporalCollator
from chicken_behavior_lab.graph.graph import TemporalSkeletonGraph


def make_sample(sample_id: str, length: int, start: int):
    features = np.ones((length, 3, 2), dtype=np.float32)
    frame_mask = np.ones(length, dtype=bool)
    node_mask = np.ones((length, 3), dtype=bool)
    if length > 1:
        node_mask[1, 2] = False
    graph = TemporalSkeletonGraph(
        node_features=features,
        edge_index=np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
        frame_mask=frame_mask,
        node_mask=node_mask,
        edge_features=np.ones((length, 4, 2), dtype=np.float32),
    )
    return GraphSample(
        graph=graph,
        label=0,
        behavior_id="feeding",
        sample_id=sample_id,
        metadata={
            "video_id": f"video_{sample_id}",
            "track_id": 1,
            "start_frame": start,
            "end_frame": start + length - 1,
        },
    )


def test_temporal_collator_pads_variable_length_samples_and_preserves_masks():
    dataset = TemporalPyGDataset([
        make_sample("short", 2, 10),
        make_sample("long", 4, 20),
    ])
    batch = TemporalCollator()([dataset[0], dataset[1]])

    assert batch.x.shape == (2, 4, 3, 2)
    assert batch.frame_mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
    assert batch.node_mask.shape == (2, 4, 3)
    assert batch.edge_mask.shape == (2, 4, 4)
    assert not batch.node_mask[0, 2:].any()
    assert not batch.edge_mask[0, 2:].any()
    assert batch.edge_attr.shape == (2, 4, 4, 2)
    assert len(batch.frame_batches) == 4
    assert batch.frame_batches[0].num_graphs == 2
    assert batch.frame_batches[0].edge_attr.shape[0] == 8
    # At t=1, the two edges incident to the invalid third node are omitted
    # from that sample's message-passing graph.
    assert batch.frame_batches[1].edge_attr.shape[0] == 4


def test_temporal_batch_to_device():
    dataset = TemporalPyGDataset([make_sample("single", 2, 0)])
    batch = TemporalCollator()([dataset[0]]).to(torch.device("cpu"))
    assert batch.x.device.type == "cpu"
    assert batch.frame_mask.device.type == "cpu"
    assert batch.node_mask.device.type == "cpu"
    assert batch.edge_mask.device.type == "cpu"
    assert batch.frame_batches[0].x.device.type == "cpu"

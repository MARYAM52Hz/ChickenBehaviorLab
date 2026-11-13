from __future__ import annotations

import numpy as np
import torch

from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.dataset.temporal_collate import make_temporal_dataloader
from chicken_behavior_lab.dataset.temporal_pyg_dataset import TemporalPyGDataset
from chicken_behavior_lab.graph.graph import TemporalSkeletonGraph
from chicken_behavior_lab.models.temporal_behavior_gnn import TemporalBehaviorGNN


def make_sample(sample_id: str, length: int, offset: int) -> GraphSample:
    rng = np.random.default_rng(offset)
    features = rng.normal(size=(length, 5, 8)).astype(np.float32)
    frame_mask = np.ones(length, dtype=bool)
    node_mask = np.ones((length, 5), dtype=bool)
    node_mask[0, 4] = False
    if length > 2:
        frame_mask[1] = False
        node_mask[1] = False
    graph = TemporalSkeletonGraph(
        node_features=features,
        edge_index=np.asarray([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=np.int64),
        frame_mask=frame_mask,
        node_mask=node_mask,
        edge_features=rng.normal(size=(length, 8, 4)).astype(np.float32),
    )
    return GraphSample(
        graph=graph,
        label=offset % 2,
        behavior_id=("feeding", "walking")[offset % 2],
        sample_id=sample_id,
        metadata={
            "video_id": f"video_{sample_id}",
            "track_id": "track_1",
            "start_frame": 0,
            "end_frame": length - 1,
        },
    )


def make_model() -> TemporalBehaviorGNN:
    return TemporalBehaviorGNN(
        node_feature_dim=8,
        edge_feature_dim=4,
        spatial_hidden_dim=16,
        temporal_hidden_dim=24,
        num_classes=2,
        num_gnn_layers=1,
        num_gru_layers=1,
        dropout=0.0,
    )


def make_batch():
    dataset = TemporalPyGDataset([
        make_sample("short", 3, 0),
        make_sample("long", 5, 1),
    ])
    loader = make_temporal_dataloader(dataset, batch_size=2, shuffle=False)
    return next(iter(loader))


def test_temporal_behavior_gnn_forward_with_padding_and_internal_gaps():
    batch = make_batch()
    assert batch.x.shape == (2, 5, 5, 8)
    assert batch.frame_mask[0].tolist() == [True, False, True, False, False]
    logits = make_model()(batch)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_temporal_behavior_gnn_backward():
    batch = make_batch()
    model = make_model()
    loss = model(batch).square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in gradients)

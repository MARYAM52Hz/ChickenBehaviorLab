from __future__ import annotations

import numpy as np
import torch

from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.dataset.temporal_builder import TemporalSequenceBuilder
from chicken_behavior_lab.dataset.temporal_collate import make_temporal_dataloader
from chicken_behavior_lab.dataset.temporal_pyg_dataset import TemporalPyGDataset
from chicken_behavior_lab.models.temporal_behavior_gnn import TemporalBehaviorGNN


class FrameGraph:
    def __init__(self, frame_index: int) -> None:
        rng = np.random.default_rng(frame_index)
        self.node_features = rng.normal(size=(13, 8)).astype(np.float32)
        self.edge_index = np.asarray(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64
        )
        self.edge_features = rng.normal(size=(4, 4)).astype(np.float32)

    def validate(self) -> None:
        if self.node_features.shape != (13, 8):
            raise ValueError("Unexpected node-feature shape.")
        if self.edge_index.shape != (2, 4):
            raise ValueError("Unexpected edge-index shape.")
        if self.edge_features.shape != (4, 4):
            raise ValueError("Unexpected edge-feature shape.")


def make_samples(number_of_frames: int = 16) -> list[GraphSample]:
    return [
        GraphSample(
            graph=FrameGraph(frame),
            label=frame % 3,
            behavior_id=("feeding", "walking", "standing")[frame % 3],
            sample_id=f"video_001_track_1_frame_{frame:04d}",
            metadata={
                "video_id": "video_001",
                "track_id": "track_1",
                "start_frame": frame,
                "end_frame": frame,
            },
        )
        for frame in range(number_of_frames)
    ]


def make_model() -> TemporalBehaviorGNN:
    return TemporalBehaviorGNN(
        node_feature_dim=8,
        edge_feature_dim=4,
        spatial_hidden_dim=16,
        temporal_hidden_dim=24,
        num_classes=3,
        num_gnn_layers=1,
        num_gru_layers=1,
        dropout=0.0,
    )


def make_batch():
    dataset = TemporalSequenceBuilder(
        sequence_length=8,
        sequence_stride=4,
    ).build(make_samples())
    pyg_dataset = TemporalPyGDataset(dataset.samples)
    loader = make_temporal_dataloader(pyg_dataset, batch_size=2, shuffle=False)
    return next(iter(loader))


def test_temporal_behavior_gnn_forward():
    batch = make_batch()
    assert batch.x.shape == (2, 8, 13, 8)
    assert len(batch.frame_batches) == 8
    assert batch.track_id == ["track_1", "track_1"]

    model = make_model()
    logits = model(batch)
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()


def test_temporal_behavior_gnn_backward():
    batch = make_batch()
    model = make_model()
    logits = model(batch)
    loss = logits.square().mean()
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)

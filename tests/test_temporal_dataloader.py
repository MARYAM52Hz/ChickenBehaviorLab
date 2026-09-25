from __future__ import annotations

from dataclasses import dataclass

import torch

from chicken_behavior_lab.dataset.temporal_builder import (
    TemporalSequenceBuilder,
)

from chicken_behavior_lab.dataset.temporal_pyg_dataset import (
    TemporalPyGDataset,
)

from chicken_behavior_lab.dataset.temporal_collate import (
    make_temporal_dataloader,
)


@dataclass
class DummySample:

    x: torch.Tensor

    edge_index: torch.Tensor

    edge_attr: torch.Tensor

    metadata: dict


def make_sample(
    frame_id: int,
) -> DummySample:

    return DummySample(
        x=torch.randn(
            13,
            8,
        ),
        edge_index=torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
            ],
            dtype=torch.long,
        ),
        edge_attr=torch.randn(
            3,
            4,
        ),
        metadata={
            "video_id": "video_001",
            "track_id": 1,
            "frame_id": frame_id,
            "behavior_id": 0,
        },
    )


def test_temporal_dataloader():

    samples = [
        make_sample(i)
        for i in range(16)
    ]

    builder = TemporalSequenceBuilder(
        sequence_length=8,
        sequence_stride=4,
    )

    windows = builder.build(
        samples
    )

    dataset = TemporalPyGDataset(
        windows
    )

    loader = make_temporal_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    batch = next(
        iter(loader)
    )

    assert batch.x.ndim == 4

    assert batch.x.shape == (
        2,
        8,
        13,
        8,
    )

    assert batch.y.shape == (
        2,
    )

    assert len(
        batch.edge_index
    ) == 8

    assert len(
        batch.edge_attr
    ) == 8

    assert batch.edge_attr[0] is not None

    assert batch.edge_attr[0].shape == (
        2,
        3,
        4,
    )


def test_edge_features_are_not_dropped():

    samples = [
        make_sample(i)
        for i in range(8)
    ]

    builder = TemporalSequenceBuilder(
        sequence_length=8,
        sequence_stride=1,
    )

    windows = builder.build(
        samples
    )

    dataset = TemporalPyGDataset(
        windows
    )

    loader = make_temporal_dataloader(
        dataset,
        batch_size=1,
    )

    batch = next(
        iter(loader)
    )

    assert batch.edge_attr[0] is not None

    assert torch.isfinite(
        batch.edge_attr[0]
    ).all()

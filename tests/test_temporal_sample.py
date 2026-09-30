from __future__ import annotations

import numpy as np
import pytest

from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)

from chicken_behavior_lab.dataset.temporal_sample import (
    TemporalGraphSample,
)


class DummyGraph:
    def __init__(self) -> None:
        self.node_features = np.zeros(
            (3, 4),
            dtype=np.float32,
        )

        self.edge_index = np.array(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=np.int64,
        )

        self.edge_features = None

    def validate(self) -> None:
        pass


def make_graph_sample(
    frame: int,
    *,
    video_id: str = "video_001",
    track_id: int = 1,
    label: int = 0,
    behavior_id: str = "feeding",
) -> GraphSample:

    return GraphSample(
        graph=DummyGraph(),
        label=label,
        behavior_id=behavior_id,
        sample_id=f"sample_{frame}",
        metadata={
            "video_id": video_id,
            "track_id": track_id,
            "start_frame": frame,
            "end_frame": frame,
        },
    )


def test_temporal_graph_sample() -> None:
    graphs = [
        make_graph_sample(0),
        make_graph_sample(1),
        make_graph_sample(2),
    ]

    sample = TemporalGraphSample(
        graphs=graphs,
        label=0,
        behavior_id="feeding",
        sample_id="temporal_001",
    )

    sample.validate()

    assert len(sample) == 3
    assert sample.sequence_length == 3
    assert sample.video_id == "video_001"
    assert sample.track_id == 1
    assert sample.first_frame == 0
    assert sample.last_frame == 2


def test_temporal_sample_rejects_empty_graphs() -> None:
    sample = TemporalGraphSample(
        graphs=[],
        label=0,
        behavior_id="feeding",
        sample_id="empty",
    )

    with pytest.raises(ValueError):
        sample.validate()


def test_temporal_sample_rejects_video_mismatch() -> None:
    sample = TemporalGraphSample(
        graphs=[
            make_graph_sample(
                0,
                video_id="video_001",
            ),
            make_graph_sample(
                1,
                video_id="video_002",
            ),
        ],
        label=0,
        behavior_id="feeding",
        sample_id="mismatch",
    )

    with pytest.raises(ValueError):
        sample.validate()

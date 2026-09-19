from types import SimpleNamespace

import numpy as np

from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)
from chicken_behavior_lab.dataset.temporal_builder import (
    TemporalSequenceBuilder,
)


def make_graph_sample(
    sample_id,
    start_frame,
    end_frame,
    behavior_id="feeding",
    label=0,
):
    graph = SimpleNamespace(
        node_features=np.zeros(
            (2, 3),
            dtype=np.float32,
        ),
        edge_index=np.empty(
            (2, 0),
            dtype=np.int64,
        ),
        edge_features=None,
        validate=lambda: None,
    )

    return GraphSample(
        graph=graph,
        label=label,
        behavior_id=behavior_id,
        sample_id=sample_id,
        metadata={
            "video_id": "video_001",
            "track_id": 1,
            "start_frame": start_frame,
            "end_frame": end_frame,
        },
    )


def test_sequence_builder():
    samples = [
        make_graph_sample(
            "sample_001",
            100,
            119,
        ),
        make_graph_sample(
            "sample_002",
            120,
            139,
        ),
        make_graph_sample(
            "sample_003",
            140,
            159,
        ),
        make_graph_sample(
            "sample_004",
            160,
            179,
        ),
    ]

    builder = TemporalSequenceBuilder(
        sequence_length=2,
        stride=1,
    )

    sequences = builder.build(
        samples
    )

    assert len(sequences) == 3

    assert len(
        sequences[0]
    ) == 2

    assert (
        sequences[0].first_frame
        == 100
    )

    assert (
        sequences[0].last_frame
        == 139
    )

    assert (
        sequences[1].first_frame
        == 120
    )

    assert (
        sequences[1].last_frame
        == 159
    )


def test_mixed_behavior_sequence_is_skipped():
    samples = [
        make_graph_sample(
            "sample_001",
            100,
            119,
            "feeding",
            0,
        ),
        make_graph_sample(
            "sample_002",
            120,
            139,
            "feeding",
            0,
        ),
        make_graph_sample(
            "sample_003",
            140,
            159,
            "standing",
            2,
        ),
        make_graph_sample(
            "sample_004",
            160,
            179,
            "standing",
            2,
        ),
    ]

    builder = TemporalSequenceBuilder(
        sequence_length=4,
        stride=1,
    )

    sequences = builder.build(
        samples
    )

    assert len(sequences) == 0

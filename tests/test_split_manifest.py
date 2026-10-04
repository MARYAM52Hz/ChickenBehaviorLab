from __future__ import annotations

import numpy as np

from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)
from chicken_behavior_lab.graph.graph import (
    TemporalSkeletonGraph,
)
from chicken_behavior_lab.io.experiment_io import (
    ExperimentOutputWriter,
)


def make_sample(
    sample_id: str,
    video_id: str,
    track_id: int,
) -> GraphSample:

    graph = TemporalSkeletonGraph(
        node_features=np.zeros(
            (3, 4),
            dtype=np.float32,
        ),
        edge_index=np.asarray(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=np.int64,
        ),
        edge_features=None,
    )

    return GraphSample(
        graph=graph,
        label=0,
        behavior_id="feeding",
        sample_id=sample_id,
        metadata={
            "video_id": video_id,
            "track_id": track_id,
            "start_frame": 0,
            "end_frame": 15,
        },
    )


def test_split_manifest_contains_sample_identity(
    tmp_path,
) -> None:

    writer = ExperimentOutputWriter(
        tmp_path
    )

    experiment_dir = (
        writer.create_experiment_directory(
            timestamp="20261004_120002"
        )
    )

    train = [
        make_sample(
            "sample_001",
            "video_001",
            1,
        )
    ]

    validation = [
        make_sample(
            "sample_002",
            "video_002",
            1,
        )
    ]

    test = [
        make_sample(
            "sample_003",
            "video_003",
            2,
        )
    ]

    path = writer.save_split_manifest(
        experiment_dir,
        train_samples=train,
        validation_samples=validation,
        test_samples=test,
        group_key="video_id",
        train_groups=["video_001"],
        validation_groups=["video_002"],
        test_groups=["video_003"],
    )

    assert path.exists()

    import json

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    assert manifest["group_key"] == "video_id"

    assert manifest["train_groups"] == [
        "video_001"
    ]

    assert manifest["train"][0][
        "sample_id"
    ] == "sample_001"

    assert manifest["test"][0][
        "video_id"
    ] == "video_003"

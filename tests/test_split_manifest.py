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
            (13, 4),
            dtype=np.float32,
        ),
        edge_index=np.array(
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
            "end_frame": 10,
        },
    )


def test_save_split_manifest(
    tmp_path,
) -> None:

    writer = ExperimentOutputWriter(
        root_directory=tmp_path
    )

    experiment_dir = (
        writer.create_experiment_directory(
            timestamp="20261005_211502"
        )
    )

    manifest = {
        "train_groups": [
            "video_001"
        ],
        "validation_groups": [
            "video_002"
        ],
        "test_groups": [
            "video_003"
        ],
        "train_samples": [
            {
                "sample_id": "sample_001",
                "video_id": "video_001",
                "track_id": 1,
            }
        ],
    }

    path = writer.save_split_manifest(
        experiment_dir,
        manifest,
    )

    assert path.exists()

    text = path.read_text(
        encoding="utf-8"
    )

    assert "video_001" in text
    assert "sample_001" in text

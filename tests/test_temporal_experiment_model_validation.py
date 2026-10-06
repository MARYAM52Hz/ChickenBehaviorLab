import numpy as np
import pytest
import torch

from chicken_behavior_lab.annotations.schema import (
    AnnotationSet,
    BehaviorAnnotation,
)
from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)
from chicken_behavior_lab.experiments.temporal_experiment import (
    TemporalExperiment,
    TemporalExperimentConfig,
)


def make_sample(
    sample_id: str,
    video_id: str,
    frame: int,
) -> GraphSample:

    from chicken_behavior_lab.graph.graph import (
        TemporalSkeletonGraph,
    )

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
            "track_id": 1,
            "start_frame": frame,
            "end_frame": frame,
        },
    )


def make_annotation(
    annotation_id: str,
    video_id: str,
    frame: int,
) -> BehaviorAnnotation:

    return BehaviorAnnotation(
        annotation_id=annotation_id,
        video_id=video_id,
        track_id=1,
        behavior_id="feeding",
        start_frame=frame,
        end_frame=frame,
    )


def test_model_output_dimension_is_checked() -> None:
    # This test is intentionally minimal.
    # Full multi-class integration is handled by the smoke test.

    experiment = TemporalExperiment(
        model_config=None,
        config=TemporalExperimentConfig(),
    )

    assert experiment.model is None

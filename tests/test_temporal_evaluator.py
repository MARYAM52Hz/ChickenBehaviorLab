from __future__ import annotations

from types import SimpleNamespace

import torch

from chicken_behavior_lab.dataset.temporal_batch import TemporalBatch
from chicken_behavior_lab.training.temporal_evaluator import (
    TemporalEvaluator,
)


class DummyTemporalModel(torch.nn.Module):
    def forward(self, batch: TemporalBatch) -> torch.Tensor:
        batch_size = batch.batch_size

        logits = torch.zeros(
            batch_size,
            3,
            device=batch.x.device,
        )

        # Predict class 0 for every sample.
        logits[:, 0] = 5.0

        return logits


def make_batch() -> TemporalBatch:
    return TemporalBatch(
        x=torch.randn(2, 4, 13, 8),
        edge_index=torch.tensor(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=torch.long,
        ),
        edge_attr=None,
        y=torch.tensor(
            [0, 1],
            dtype=torch.long,
        ),
        sample_id=[
            "sample_001",
            "sample_002",
        ],
        video_id=[
            "video_001",
            "video_001",
        ],
        track_id=torch.tensor(
            [10, 10],
            dtype=torch.long,
        ),
        start_frame=torch.tensor(
            [100, 104],
            dtype=torch.long,
        ),
        end_frame=torch.tensor(
            [103, 107],
            dtype=torch.long,
        ),
        behavior_id=[
            "feeding",
            "walking",
        ],
    )


def test_temporal_evaluator() -> None:
    model = DummyTemporalModel()

    evaluator = TemporalEvaluator(
        model=model,
        device="cpu",
        index_to_label={
            0: "feeding",
            1: "walking",
            2: "standing",
        },
    )

    class DummyLoader:
        def __iter__(self):
            yield make_batch()

    result = evaluator.evaluate(DummyLoader())

    assert result.y_true == [0, 1]
    assert result.y_pred == [0, 0]

    assert result.metrics["num_samples"] == 2
    assert result.metrics["accuracy"] == 0.5

    assert len(result.prediction_records) == 2

    first = result.prediction_records[0]

    assert first.sample_id == "sample_001"
    assert first.video_id == "video_001"
    assert first.track_id == 10
    assert first.start_frame == 100
    assert first.end_frame == 103
    assert first.true_index == 0
    assert first.predicted_index == 0

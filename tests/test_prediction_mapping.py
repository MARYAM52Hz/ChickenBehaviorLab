from dataclasses import dataclass

from chicken_behavior_lab.analysis import (
    PredictionErrorAnalyzer,
)


@dataclass
class MockRawPrediction:
    sample_id: str
    video_id: str
    track_id: int
    start_frame: int
    end_frame: int
    true_index: int
    predicted_index: int
    confidence: float


@dataclass
class MockEvaluationResult:
    y_true: list[int]
    y_pred: list[int]
    prediction_records: list


def test_index_to_behavior_mapping():
    result = MockEvaluationResult(
        y_true=[0, 1],
        y_pred=[1, 1],
        prediction_records=[
            MockRawPrediction(
                sample_id="sample_001",
                video_id="video_001",
                track_id=1,
                start_frame=100,
                end_frame=150,
                true_index=0,
                predicted_index=1,
                confidence=0.82,
            ),
            MockRawPrediction(
                sample_id="sample_002",
                video_id="video_001",
                track_id=2,
                start_frame=200,
                end_frame=250,
                true_index=1,
                predicted_index=1,
                confidence=0.91,
            ),
        ],
    )

    analyzer = (
        PredictionErrorAnalyzer
        .from_evaluation_result(
            evaluation_result=result,
            index_to_label={
                0: "feeding",
                1: "standing",
            },
        )
    )

    assert len(analyzer) == 2

    assert (
        analyzer.predictions[0]
        .true_behavior
        == "feeding"
    )

    assert (
        analyzer.predictions[0]
        .predicted_behavior
        == "standing"
    )

    assert (
        analyzer.predictions[1]
        .true_behavior
        == "standing"
    )

    assert (
        analyzer.predictions[1]
        .predicted_behavior
        == "standing"
    )

    assert (
        analyzer.predictions[0]
        .confidence
        == 0.82
    )

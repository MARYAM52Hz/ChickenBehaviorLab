from chicken_behavior_lab.analysis import (
    PredictionErrorAnalyzer,
    PredictionRecord,
)


def make_predictions():
    return [
        PredictionRecord(
            sample_id="sample_001",
            video_id="video_001",
            track_id=1,
            start_frame=100,
            end_frame=180,
            true_behavior="feeding",
            predicted_behavior="feeding",
            confidence=0.94,
        ),
        PredictionRecord(
            sample_id="sample_002",
            video_id="video_001",
            track_id=1,
            start_frame=181,
            end_frame=230,
            true_behavior="feeding",
            predicted_behavior="standing",
            confidence=0.61,
        ),
        PredictionRecord(
            sample_id="sample_003",
            video_id="video_001",
            track_id=2,
            start_frame=100,
            end_frame=160,
            true_behavior="walking",
            predicted_behavior="standing",
            confidence=0.87,
        ),
        PredictionRecord(
            sample_id="sample_004",
            video_id="video_002",
            track_id=4,
            start_frame=50,
            end_frame=90,
            true_behavior="standing",
            predicted_behavior="standing",
            confidence=0.91,
        ),
    ]


def test_prediction_record_correctness():
    prediction = PredictionRecord(
        sample_id="sample_001",
        video_id="video_001",
        track_id=1,
        start_frame=1,
        end_frame=10,
        true_behavior="feeding",
        predicted_behavior="feeding",
        confidence=0.90,
    )

    assert prediction.is_correct
    assert not prediction.is_error


def test_prediction_record_error():
    prediction = PredictionRecord(
        sample_id="sample_001",
        video_id="video_001",
        track_id=1,
        start_frame=1,
        end_frame=10,
        true_behavior="feeding",
        predicted_behavior="standing",
        confidence=0.60,
    )

    assert not prediction.is_correct
    assert prediction.is_error


def test_accuracy_and_error_rate():
    analyzer = PredictionErrorAnalyzer(
        make_predictions()
    )

    assert len(analyzer) == 4
    assert len(analyzer.errors) == 2

    assert analyzer.accuracy() == 0.5
    assert analyzer.error_rate() == 0.5


def test_low_confidence_errors():
    analyzer = PredictionErrorAnalyzer(
        make_predictions()
    )

    errors = (
        analyzer.low_confidence_errors(
            threshold=0.70
        )
    )

    assert len(errors) == 1
    assert (
        errors[0].sample_id
        == "sample_002"
    )


def test_high_confidence_errors():
    analyzer = PredictionErrorAnalyzer(
        make_predictions()
    )

    errors = (
        analyzer.high_confidence_errors(
            threshold=0.80
        )
    )

    assert len(errors) == 1
    assert (
        errors[0].sample_id
        == "sample_003"
    )


def test_confusion_pairs():
    analyzer = PredictionErrorAnalyzer(
        make_predictions()
    )

    pairs = analyzer.confusion_pairs()

    assert len(pairs) == 2

    assert (
        pairs[0]["true_behavior"]
        == "feeding"
    )

    assert (
        pairs[0]["predicted_behavior"]
        == "standing"
    )

    assert pairs[0]["count"] == 1


def test_error_rate_by_behavior():
    analyzer = PredictionErrorAnalyzer(
        make_predictions()
    )

    results = (
        analyzer.error_rate_by_true_behavior()
    )

    feeding = next(
        item
        for item in results
        if item["behavior"] == "feeding"
    )

    walking = next(
        item
        for item in results
        if item["behavior"] == "walking"
    )

    standing = next(
        item
        for item in results
        if item["behavior"] == "standing"
    )

    assert feeding["total"] == 2
    assert feeding["errors"] == 1
    assert feeding["error_rate"] == 0.5

    assert walking["total"] == 1
    assert walking["errors"] == 1
    assert walking["error_rate"] == 1.0

    assert standing["total"] == 1
    assert standing["errors"] == 0
    assert standing["error_rate"] == 0.0


def test_save_and_load_json(tmp_path):
    analyzer = PredictionErrorAnalyzer(
        make_predictions()
    )

    path = (
        tmp_path
        / "prediction_errors.json"
    )

    analyzer.save_json(path)

    loaded = (
        PredictionErrorAnalyzer.load_json(
            path
        )
    )

    assert len(loaded) == 4
    assert loaded.accuracy() == 0.5
    assert len(loaded.errors) == 2

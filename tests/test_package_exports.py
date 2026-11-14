def test_training_exports():
    from chicken_behavior_lab.training import (
        ClassificationLoss,
        Trainer,
        Evaluator,
        EvaluationResult,
        RawPredictionRecord,
        load_checkpoint,
        ExperimentResultManager,
        TemporalEvaluator,
        TemporalEvaluationResult,
        TemporalPredictionRecord,
    )

    assert ClassificationLoss is not None
    assert Trainer is not None
    assert Evaluator is not None
    assert EvaluationResult is not None
    assert RawPredictionRecord is not None
    assert load_checkpoint is not None
    assert ExperimentResultManager is not None

    assert TemporalEvaluator is not None
    assert TemporalEvaluationResult is not None
    assert TemporalPredictionRecord is not None


def test_analysis_exports():
    from chicken_behavior_lab.analysis import (
        TrainingHistoryAnalyzer,
        ConfusionMatrixAnalyzer,
        PredictionRecord,
        PredictionErrorAnalyzer,
        TemporalPredictionError,
        TemporalPredictionErrorAnalyzer,
    )

    assert TrainingHistoryAnalyzer is not None
    assert ConfusionMatrixAnalyzer is not None
    assert PredictionRecord is not None
    assert PredictionErrorAnalyzer is not None

    assert TemporalPredictionError is not None
    assert TemporalPredictionErrorAnalyzer is not None


def test_all_public_analysis_exports_resolve():
    import chicken_behavior_lab.analysis as analysis

    expected_exports = {
        "TrainingHistoryAnalyzer",
        "TrainingCurveData",
        "ConfusionMatrixAnalyzer",
        "PredictionRecord",
        "PredictionErrorAnalyzer",
        "FeatureDiagnosticRecord",
        "FeatureDiagnosticAnalyzer",
        "GraphFeatureDiagnosticExtractor",
        "TemporalPredictionError",
        "TemporalPredictionErrorAnalyzer",
    }

    assert expected_exports.issubset(set(analysis.__all__))
    for name in expected_exports:
        assert getattr(analysis, name) is not None

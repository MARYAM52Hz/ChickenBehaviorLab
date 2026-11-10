"""Analysis utilities, imported lazily to keep optional dependencies isolated."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "TrainingHistoryAnalyzer": (".training_curves", "TrainingHistoryAnalyzer"),
    "TrainingCurveData": (".training_curves", "TrainingCurveData"),
    "ConfusionMatrixAnalyzer": (".confusion_matrix", "ConfusionMatrixAnalyzer"),
    "PredictionRecord": (".error_analysis", "PredictionRecord"),
    "PredictionErrorAnalyzer": (".error_analysis", "PredictionErrorAnalyzer"),
    "TemporalPredictionError": (".temporal_error_analysis", "TemporalPredictionError"),
    "TemporalPredictionErrorAnalyzer": (".temporal_error_analysis", "TemporalPredictionErrorAnalyzer"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Resolve public analysis exports when requested."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value

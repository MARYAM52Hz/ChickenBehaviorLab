"""Training utilities with lazy loading of PyTorch-dependent exports."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "ClassificationLoss": (".losses", "ClassificationLoss"),
    "Trainer": (".trainer", "Trainer"),
    "build_trainer": (".trainer_factory", "build_trainer"),
    "CheckpointManager": (".checkpoint", "CheckpointManager"),
    "load_checkpoint": (".checkpoint", "load_checkpoint"),
    "Evaluator": (".evaluator", "Evaluator"),
    "EvaluationResult": (".evaluator", "EvaluationResult"),
    "RawPredictionRecord": (".evaluator", "RawPredictionRecord"),
    "ExperimentResultManager": (".experiment_results", "ExperimentResultManager"),
    "TemporalEvaluator": (".temporal_evaluator", "TemporalEvaluator"),
    "TemporalEvaluationResult": (".temporal_evaluator", "TemporalEvaluationResult"),
    "TemporalPredictionRecord": (".temporal_evaluator", "TemporalPredictionRecord"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Load a public training symbol only when requested by a caller."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value

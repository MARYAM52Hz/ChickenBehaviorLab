"""Model exports with lazy loading for optional deep-learning dependencies.

Importing lightweight data models such as Detection, Keypoint, or Skeleton must
not import PyTorch. Neural-network classes are loaded only when requested.
"""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "ChickenBehaviorGNN": (
        "chicken_behavior_lab.models.chicken_behavior_gnn",
        "ChickenBehaviorGNN",
    ),
    "SpatialGraphEncoder": (
        "chicken_behavior_lab.models.spatial_encoder",
        "SpatialGraphEncoder",
    ),
    "GRUTemporalEncoder": (
        "chicken_behavior_lab.models.temporal_encoder",
        "GRUTemporalEncoder",
    ),
    "TemporalBehaviorGNN": (
        "chicken_behavior_lab.models.temporal_behavior_gnn",
        "TemporalBehaviorGNN",
    ),
    "TemporalModelConfig": (
        "chicken_behavior_lab.models.factory",
        "TemporalModelConfig",
    ),
    "build_temporal_model": (
        "chicken_behavior_lab.models.factory",
        "build_temporal_model",
    ),
    "build_temporal_model_from_dataset": (
        "chicken_behavior_lab.models.factory",
        "build_temporal_model_from_dataset",
    ),
    "infer_temporal_model_dimensions": (
        "chicken_behavior_lab.models.factory",
        "infer_temporal_model_dimensions",
    ),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    """Load a declared model export only when it is accessed."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

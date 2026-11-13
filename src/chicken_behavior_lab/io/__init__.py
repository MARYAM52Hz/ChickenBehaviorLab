"""Public I/O exports, loaded lazily to keep lightweight readers optional-dependency free."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "load_annotations": ("chicken_behavior_lab.io.annotation_io", "load_annotations"),
    "save_temporal_evaluation": ("chicken_behavior_lab.io.evaluation_io", "save_temporal_evaluation"),
    "ExperimentOutputWriter": ("chicken_behavior_lab.io.experiment_io", "ExperimentOutputWriter"),
    "GraphJsonLoader": ("chicken_behavior_lab.io.graph_json", "GraphJsonLoader"),
    "load_graph_samples": ("chicken_behavior_lab.io.graph_json", "load_graph_samples"),
    "MediaManifestIO": ("chicken_behavior_lab.io.media_manifest_io", "MediaManifestIO"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load an I/O helper only when the caller requests that export."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

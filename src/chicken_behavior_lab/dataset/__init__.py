"""Dataset interfaces with lazy loading of optional PyTorch components.

Lightweight sample and grouping utilities can be imported without importing
PyTorch or PyTorch Geometric. Training-specific adapters are loaded on demand.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "GraphSample": (".sample", "GraphSample"),
    "DatasetFactory": (".factory", "DatasetFactory"),
    "AnnotationRecord": (".factory", "AnnotationRecord"),
    "GraphDataset": (".graph_dataset", "GraphDataset"),
    "PyGGraphDataset": (".pyg_dataset", "PyGGraphDataset"),
    "FrameGraphDataset": (".pyg_dataset", "FrameGraphDataset"),
    "TemporalGraphSample": (".temporal_sample", "TemporalGraphSample"),
    "TemporalGraphDataset": (".temporal_dataset", "TemporalGraphDataset"),
    "TemporalSequenceBuilder": (".temporal_builder", "TemporalSequenceBuilder"),
    "TemporalPyGDataset": (".temporal_pyg_dataset", "TemporalPyGDataset"),
    "TemporalBatch": (".temporal_batch", "TemporalBatch"),
    "TemporalCollator": (".temporal_collate", "TemporalCollator"),
    "make_temporal_dataloader": (".temporal_collate", "make_temporal_dataloader"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Resolve public dataset exports only when a caller requests them."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value

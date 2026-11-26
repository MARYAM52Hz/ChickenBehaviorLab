"""Shared, named edge-feature contract for both graph representations."""

from __future__ import annotations

import numpy as np

EDGE_FEATURE_NAMES = (
    "relative_dx",
    "relative_dy",
    "distance",
    "delta_vx",
    "delta_vy",
    "delta_ax",
    "delta_ay",
    "source_confidence",
    "target_confidence",
)

_DEFAULT_FEATURE_NAMES = ("x", "y", "vx", "vy", "ax", "ay", "confidence")
_LEGACY_XY_NAMES = ("x", "y")


def resolve_feature_names(feature_dim: int, feature_names: tuple[str, ...] | None) -> tuple[str, ...]:
    """Resolve named node channels while retaining legacy XY/7-channel inputs."""
    if feature_names is None:
        if feature_dim == len(_DEFAULT_FEATURE_NAMES):
            return _DEFAULT_FEATURE_NAMES
        if feature_dim == 2:
            return _LEGACY_XY_NAMES
        raise ValueError(
            "feature_names are required unless node features use the legacy "
            "XY (2-channel) or canonical 7-channel layout."
        )
    names = tuple(feature_names)
    if len(names) != feature_dim:
        raise ValueError("feature_names length must match node feature dimension.")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique.")
    if "x" not in names or "y" not in names:
        raise ValueError("feature_names must include x and y.")
    return names


def build_combined_edge_features(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    feature_names: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Build 9 numeric edge attributes for (T,V,F) or flattened (N,F) nodes.

    Order: dx, dy, distance, dvx, dvy, dax, day, source confidence,
    target confidence. Missing optional motion/confidence channels are zero.
    Edge validity is deliberately represented by a separate mask.
    """
    nodes = np.asarray(node_features, dtype=np.float32)
    edges = np.asarray(edge_index, dtype=np.int64)
    if nodes.ndim not in (2, 3):
        raise ValueError("node_features must have shape (N,F) or (T,V,F).")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2,E).")

    num_nodes = nodes.shape[-2]
    if edges.size and (np.any(edges < 0) or np.any(edges >= num_nodes)):
        raise ValueError("edge_index contains an invalid node index.")

    names = resolve_feature_names(nodes.shape[-1], feature_names)
    indices = {name: names.index(name) for name in names}

    source = nodes[..., edges[0], :] if nodes.ndim == 3 else nodes[edges[0], :]
    target = nodes[..., edges[1], :] if nodes.ndim == 3 else nodes[edges[1], :]

    dx = target[..., indices["x"]] - source[..., indices["x"]]
    dy = target[..., indices["y"]] - source[..., indices["y"]]
    distance = np.sqrt(dx * dx + dy * dy)
    values = [
        dx, dy, distance,
        target[..., indices["vx"]] - source[..., indices["vx"]] if "vx" in indices else np.zeros_like(dx),
        target[..., indices["vy"]] - source[..., indices["vy"]] if "vy" in indices else np.zeros_like(dx),
        target[..., indices["ax"]] - source[..., indices["ax"]] if "ax" in indices else np.zeros_like(dx),
        target[..., indices["ay"]] - source[..., indices["ay"]] if "ay" in indices else np.zeros_like(dx),
        source[..., indices["confidence"]] if "confidence" in indices else np.zeros_like(dx),
        target[..., indices["confidence"]] if "confidence" in indices else np.zeros_like(dx),
    ]
    return np.stack(values, axis=-1).astype(np.float32, copy=False)

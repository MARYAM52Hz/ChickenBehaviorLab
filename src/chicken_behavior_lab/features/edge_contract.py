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


def validate_declared_edge_contract(
    *,
    node_features: np.ndarray,
    edge_index: np.ndarray,
    edge_features: np.ndarray | None,
    edge_feature_names: tuple[str, ...] | list[str] | None,
    node_feature_names: tuple[str, ...] | list[str] | None = None,
    frame_mask: np.ndarray | None = None,
    node_mask: np.ndarray | None = None,
) -> None:
    """Validate explicitly declared combined-edge metadata at ingestion.

    Files without edge_feature_names are treated as legacy payloads and
    remain loadable. Once names are declared, the complete nine-channel
    contract and canonical bidirectional topology are mandatory.
    """
    if edge_feature_names is None:
        return

    from chicken_behavior_lab.skeleton.topology import canonicalize_edge_index

    names = tuple(str(name) for name in edge_feature_names)
    if names != EDGE_FEATURE_NAMES:
        raise ValueError(
            "Declared edge_feature_names must match the canonical nine-channel "
            f"contract exactly: {EDGE_FEATURE_NAMES}."
        )

    nodes = np.asarray(node_features)
    edges = np.asarray(edge_index, dtype=np.int64)
    attributes = (
        None if edge_features is None
        else np.asarray(edge_features, dtype=np.float32)
    )
    if nodes.ndim != 3:
        raise ValueError("Declared combined edge contract requires node_features (T,V,F).")
    expected = (nodes.shape[0], edges.shape[1], len(EDGE_FEATURE_NAMES))
    if attributes is None or attributes.shape != expected:
        raise ValueError(
            f"Declared combined edge contract requires edge_features shape {expected}."
        )
    if not np.isfinite(attributes).all():
        raise ValueError("Declared combined edge features must contain only finite values.")

    expected_attributes = build_combined_edge_features(
        nodes, edges,
        None if node_feature_names is None else tuple(node_feature_names),
    )

    # Match the graph builders: an edge is usable only when its frame and
    # both endpoint keypoints are valid. Invalid edge attributes are stored
    # as zeros, while validity remains represented by the masks.
    if frame_mask is not None:
        frame_mask = np.asarray(frame_mask, dtype=bool)
        if frame_mask.shape != (nodes.shape[0],):
            raise ValueError("frame_mask must have shape (T,).")
    else:
        frame_mask = np.ones((nodes.shape[0],), dtype=bool)

    if node_mask is not None:
        node_mask = np.asarray(node_mask, dtype=bool)
        if node_mask.shape != nodes.shape[:2]:
            raise ValueError("node_mask must have shape (T, V).")
    else:
        node_mask = np.ones(nodes.shape[:2], dtype=bool)

    edge_valid = (
        frame_mask[:, None]
        & node_mask[:, edges[0]]
        & node_mask[:, edges[1]]
    )
    expected_attributes[~edge_valid] = 0.0

    if not np.allclose(attributes, expected_attributes, rtol=1e-5, atol=1e-6):
        raise ValueError(
            "Declared edge_features do not match the canonical combined "
            "features computed from node_features."
        )

    canonical = canonicalize_edge_index(
        edges, num_keypoints=nodes.shape[1], bidirectional=True
    )
    if not np.array_equal(edges, canonical):
        raise ValueError(
            "Declared combined edge contract requires canonical, deduplicated, "
            "bidirectional edge_index ordering."
        )


def validate_edge_index_input(edge_index: np.ndarray) -> np.ndarray:
    """Validate raw edge indices before converting them to an integer dtype."""
    raw = np.asarray(edge_index)
    if raw.ndim != 2 or raw.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E).")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("edge_index must contain integer-valued indices.")
    return raw.astype(np.int64, copy=False)


def validate_binary_mask(mask: np.ndarray, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Accept boolean masks or numeric masks containing only zero and one."""
    raw = np.asarray(mask)
    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if raw.dtype != np.bool_:
        if not np.issubdtype(raw.dtype, np.number):
            raise ValueError(f"{name} must contain boolean or binary numeric values.")
        if not np.isfinite(raw).all() or not np.isin(raw, (0, 1)).all():
            raise ValueError(f"{name} must contain only boolean or 0/1 values.")
    return raw.astype(bool, copy=False)


def validate_node_feature_array(node_features: np.ndarray) -> np.ndarray:
    """Validate temporal node features before downstream graph construction."""
    nodes = np.asarray(node_features, dtype=np.float32)
    if nodes.ndim != 3:
        raise ValueError("node_features must have shape (T, V, F).")
    if any(size == 0 for size in nodes.shape):
        raise ValueError("node_features must have non-zero T, V, and F dimensions.")
    if not np.isfinite(nodes).all():
        raise ValueError("node_features must contain only finite values.")
    return nodes

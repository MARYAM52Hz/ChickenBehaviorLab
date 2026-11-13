"""Convert skeleton frames into a deterministic numeric graph representation."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ContractValidationError, SkeletonFrame
from .keypoint_schema import KeypointSchema


@dataclass(frozen=True)
class SkeletonGraphData:
    """Numeric graph payload for one skeleton frame.

    node_features columns:
        x, y, confidence, visible

    edge_index:
        (source_indices, target_indices), each of length E.
        This is a lightweight representation, not a PyG Data object.
    """

    track_id: str
    frame_index: int
    node_names: tuple[str, ...]
    node_features: tuple[tuple[float, float, float, float], ...]
    edge_index: tuple[tuple[int, ...], tuple[int, ...]]
    feature_names: tuple[str, ...] = (
        "x",
        "y",
        "confidence",
        "visible",
    )

    def __post_init__(self) -> None:
        if len(self.node_names) != len(self.node_features):
            raise ContractValidationError(
                "node_names and node_features must have equal lengths."
            )

        if len(set(self.node_names)) != len(self.node_names):
            raise ContractValidationError(
                "node_names must be unique."
            )

        if len(self.edge_index) != 2:
            raise ContractValidationError(
                "edge_index must contain source and target index rows."
            )

        sources, targets = self.edge_index

        if len(sources) != len(targets):
            raise ContractValidationError(
                "Source and target edge counts must match."
            )

        node_count = len(self.node_names)

        for source, target in zip(sources, targets):
            if not 0 <= source < node_count:
                raise ContractValidationError(
                    f"Invalid source node index: {source}."
                )

            if not 0 <= target < node_count:
                raise ContractValidationError(
                    f"Invalid target node index: {target}."
                )

            if source == target:
                raise ContractValidationError(
                    "Self-loop edges are not allowed."
                )

        for row in self.node_features:
            if len(row) != len(self.feature_names):
                raise ContractValidationError(
                    "Each node feature row must match feature_names."
                )


class SkeletonGraphEncoder:
    """Encode a SkeletonFrame into a deterministic numeric graph payload."""

    def __init__(self, schema: KeypointSchema) -> None:
        if not isinstance(schema, KeypointSchema):
            raise ContractValidationError(
                "schema must be a KeypointSchema instance."
            )

        self.schema = schema
        self.node_index = schema.index_by_name

    def encode(self, skeleton: SkeletonFrame) -> SkeletonGraphData:
        """Encode nodes and edges without changing coordinate units."""

        if not isinstance(skeleton, SkeletonFrame):
            raise ContractValidationError(
                "skeleton must be a SkeletonFrame instance."
            )

        expected = set(self.schema.names)
        actual = set(skeleton.nodes)

        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)

            raise ContractValidationError(
                f"Skeleton/schema mismatch; missing={missing}, extra={extra}."
            )

        if skeleton.edges != self.schema.edges:
            raise ContractValidationError(
                "Skeleton edges do not match the canonical schema."
            )

        rows = []

        for name in self.schema.names:
            point = skeleton.nodes[name]

            # Visibility is explicit so masked coordinates are not
            # mistaken for observed positions by downstream feature code.
            rows.append(
                (
                    float(point.x),
                    float(point.y),
                    float(point.confidence),
                    float(point.visible),
                )
            )

        sources = tuple(
            self.node_index[source]
            for source, _ in self.schema.edges
        )
        targets = tuple(
            self.node_index[target]
            for _, target in self.schema.edges
        )

        return SkeletonGraphData(
            track_id=skeleton.track_id,
            frame_index=skeleton.frame_index,
            node_names=self.schema.names,
            node_features=tuple(rows),
            edge_index=(sources, targets),
        )

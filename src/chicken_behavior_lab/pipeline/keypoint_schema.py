"""Canonical keypoint schema for ChickenBehaviorLab."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ContractValidationError


@dataclass(frozen=True)
class KeypointSchema:
    """Define a stable order of keypoints and skeleton edges."""

    names: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.names, tuple) or not self.names:
            raise ContractValidationError(
                "Keypoint names must be a non-empty tuple."
            )

        if any(
            not isinstance(name, str) or not name.strip()
            for name in self.names
        ):
            raise ContractValidationError(
                "Every keypoint name must be a non-empty string."
            )

        if len(set(self.names)) != len(self.names):
            raise ContractValidationError(
                "Keypoint names must be unique."
            )

        known_names = set(self.names)
        normalized_edges: list[tuple[str, str]] = []
        seen_edges: set[tuple[str, str]] = set()

        if not isinstance(self.edges, tuple):
            raise ContractValidationError(
                "Edges must be provided as a tuple."
            )

        for edge in self.edges:
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                raise ContractValidationError(
                    "Each edge must contain exactly two keypoint names."
                )

            source, target = edge

            if not isinstance(source, str) or not isinstance(target, str):
                raise ContractValidationError(
                    "Edge endpoints must be strings."
                )

            if source not in known_names or target not in known_names:
                raise ContractValidationError(
                    f"Edge {edge!r} references an unknown keypoint."
                )

            if source == target:
                raise ContractValidationError(
                    "Self-loop edges are not allowed."
                )

            normalized = (source, target)

            if normalized in seen_edges:
                raise ContractValidationError(
                    f"Duplicate edge found: {normalized!r}."
                )

            seen_edges.add(normalized)
            normalized_edges.append(normalized)

        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(
            self,
            "edges",
            tuple(normalized_edges),
        )

    @property
    def index_by_name(self) -> dict[str, int]:
        """Return the canonical node index for each keypoint name."""
        return {
            name: index
            for index, name in enumerate(self.names)
        }

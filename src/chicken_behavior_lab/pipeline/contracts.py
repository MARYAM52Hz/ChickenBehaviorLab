
"""Validated data contracts between ChickenBehaviorLab pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping


class ContractValidationError(ValueError):
    """Raised when a pipeline payload violates its data contract."""


def _validate_frame_index(value: int, field_name: str = "frame_index") -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ContractValidationError(
            f"{field_name} must be a non-negative integer."
        )


def _validate_track_id(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ContractValidationError(
            "track_id must be a non-negative integer."
        )


def _finite_number(value: Real, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ContractValidationError(f"{field_name} must be numeric.")

    result = float(value)

    if not math.isfinite(result):
        raise ContractValidationError(f"{field_name} must be finite.")

    return result


def _validate_confidence(value: Real, field_name: str = "confidence") -> float:
    result = _finite_number(value, field_name)

    if not 0.0 <= result <= 1.0:
        raise ContractValidationError(
            f"{field_name} must be between 0 and 1."
        )

    return result


@dataclass(frozen=True)
class Detection:
    """One object detection in a single video frame.

    Bounding-box coordinates use the xyxy convention:
    (x_min, y_min, x_max, y_max), in image-coordinate units.
    """

    frame_index: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str | None = None

    def __post_init__(self) -> None:
        _validate_frame_index(self.frame_index)

        if len(self.bbox_xyxy) != 4:
            raise ContractValidationError(
                "bbox_xyxy must contain exactly four coordinates."
            )

        coordinates = tuple(
            _finite_number(value, "bbox_xyxy coordinate")
            for value in self.bbox_xyxy
        )

        x_min, y_min, x_max, y_max = coordinates

        if x_max <= x_min or y_max <= y_min:
            raise ContractValidationError(
                "Bounding box must have positive width and height."
            )

        _validate_confidence(self.confidence)

        if (
            isinstance(self.class_id, bool)
            or not isinstance(self.class_id, Integral)
            or self.class_id < 0
        ):
            raise ContractValidationError(
                "class_id must be a non-negative integer."
            )

        if self.class_name is not None and (
            not isinstance(self.class_name, str)
            or not self.class_name.strip()
        ):
            raise ContractValidationError(
                "class_name must be a non-empty string or None."
            )

        object.__setattr__(self, "bbox_xyxy", coordinates)
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "class_id", int(self.class_id))


@dataclass(frozen=True)
class TrackObservation:
    """A detection associated with a tracker-assigned track ID."""

    track_id: int
    detection: Detection

    def __post_init__(self) -> None:
        _validate_track_id(self.track_id)

        if not isinstance(self.detection, Detection):
            raise ContractValidationError(
                "detection must be a Detection instance."
            )

    @property
    def frame_index(self) -> int:
        return self.detection.frame_index

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        return self.detection.bbox_xyxy


@dataclass(frozen=True)
class Keypoint:
    """One image-space body keypoint."""

    x: float
    y: float
    confidence: float
    visible: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_number(self.x, "keypoint.x"))
        object.__setattr__(self, "y", _finite_number(self.y, "keypoint.y"))
        object.__setattr__(
            self,
            "confidence",
            _validate_confidence(self.confidence, "keypoint.confidence"),
        )

        if not isinstance(self.visible, bool):
            raise ContractValidationError("visible must be a boolean.")


@dataclass(frozen=True)
class PoseEstimate:
    """Named keypoints estimated for one tracked subject in one frame."""

    track_id: int
    frame_index: int
    keypoints: Mapping[str, Keypoint]

    def __post_init__(self) -> None:
        _validate_track_id(self.track_id)
        _validate_frame_index(self.frame_index)

        if not isinstance(self.keypoints, Mapping) or not self.keypoints:
            raise ContractValidationError(
                "keypoints must be a non-empty mapping."
            )

        copied: dict[str, Keypoint] = {}

        for name, point in self.keypoints.items():
            if not isinstance(name, str) or not name.strip():
                raise ContractValidationError(
                    "Keypoint names must be non-empty strings."
                )

            if not isinstance(point, Keypoint):
                raise ContractValidationError(
                    f"Keypoint {name!r} must be a Keypoint instance."
                )

            copied[name] = point

        object.__setattr__(
            self,
            "keypoints",
            MappingProxyType(copied),
        )


@dataclass(frozen=True)
class SkeletonFrame:
    """Named skeleton nodes and edges for one tracked subject in one frame.

    Edge endpoints refer to keys in ``nodes``. Edge direction is preserved;
    downstream code should define whether its graph treats edges as directed.
    """

    track_id: int
    frame_index: int
    nodes: Mapping[str, Keypoint]
    edges: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _validate_track_id(self.track_id)
        _validate_frame_index(self.frame_index)

        if not isinstance(self.nodes, Mapping) or not self.nodes:
            raise ContractValidationError(
                "nodes must be a non-empty mapping."
            )

        copied_nodes: dict[str, Keypoint] = {}

        for name, point in self.nodes.items():
            if not isinstance(name, str) or not name.strip():
                raise ContractValidationError(
                    "Skeleton node names must be non-empty strings."
                )

            if not isinstance(point, Keypoint):
                raise ContractValidationError(
                    f"Node {name!r} must be a Keypoint instance."
                )

            copied_nodes[name] = point

        normalized_edges: list[tuple[str, str]] = []
        seen_edges: set[tuple[str, str]] = set()

        for edge in self.edges:
            if (
                not isinstance(edge, (tuple, list))
                or len(edge) != 2
            ):
                raise ContractValidationError(
                    "Each edge must contain exactly two node names."
                )

            source, target = edge

            if source not in copied_nodes or target not in copied_nodes:
                raise ContractValidationError(
                    f"Edge {edge!r} references an unknown node."
                )

            if source == target:
                raise ContractValidationError(
                    "Self-loop edges are not allowed by this contract."
                )

            normalized = (source, target)

            if normalized in seen_edges:
                raise ContractValidationError(
                    f"Duplicate edge found: {normalized!r}."
                )

            seen_edges.add(normalized)
            normalized_edges.append(normalized)

        object.__setattr__(
            self,
            "nodes",
            MappingProxyType(copied_nodes),
        )
        object.__setattr__(
            self,
            "edges",
            tuple(normalized_edges),
        )


def validate_pose_matches_track(
    track: TrackObservation,
    pose: PoseEstimate,
) -> None:
    """Ensure that a pose belongs to the same track and frame as a detection."""

    if track.track_id != pose.track_id:
        raise ContractValidationError(
            "Track ID mismatch between tracking and pose estimation."
        )

    if track.frame_index != pose.frame_index:
        raise ContractValidationError(
            "Frame index mismatch between tracking and pose estimation."
        )


def validate_skeleton_matches_pose(
    pose: PoseEstimate,
    skeleton: SkeletonFrame,
) -> None:
    """Ensure that the skeleton belongs to the same subject and frame."""

    if pose.track_id != skeleton.track_id:
        raise ContractValidationError(
            "Track ID mismatch between pose and skeleton."
        )

    if pose.frame_index != skeleton.frame_index:
        raise ContractValidationError(
            "Frame index mismatch between pose and skeleton."
        )

    missing_nodes = set(skeleton.nodes) - set(pose.keypoints)

    if missing_nodes:
        raise ContractValidationError(
            "Skeleton contains nodes absent from the pose estimate: "
            f"{sorted(missing_nodes)}."
        )

"""
Chicken Behavior Annotation Standard (CBAS)
============================================

Reference implementation of the Chicken Behavior Annotation
Standard used by ChickenBehaviorLab.

CBAS defines the canonical representation of:

- Chicken identity
- Anatomical keypoints
- Skeleton topology
- Detection annotations
- Pose annotations
- Behavior annotations
- Temporal events
- Annotation metadata
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# =========================================================
# CBAS Version
# =========================================================

CBAS_VERSION = "1.0.0"


# =========================================================
# Annotation Types
# =========================================================

class AnnotationType(str, Enum):
    """Supported CBAS annotation types."""

    DETECTION = "detection"
    POSE = "pose"
    SKELETON = "skeleton"
    BEHAVIOR = "behavior"
    EVENT = "event"


# =========================================================
# Keypoint Definition
# =========================================================

@dataclass(frozen=True, slots=True)
class CBASKeypoint:
    """
    Canonical anatomical keypoint definition.
    """

    name: str

    index: int

    description: str


# =========================================================
# Canonical Chicken Skeleton
# =========================================================

CBAS_KEYPOINTS: tuple[CBASKeypoint, ...] = (
    CBASKeypoint(
        name="beak",
        index=0,
        description="Chicken beak tip.",
    ),
    CBASKeypoint(
        name="head",
        index=1,
        description="Approximate center of the head.",
    ),
    CBASKeypoint(
        name="neck",
        index=2,
        description="Approximate neck landmark.",
    ),
    CBASKeypoint(
        name="body_center",
        index=3,
        description="Approximate center of the torso.",
    ),
    CBASKeypoint(
        name="left_wing",
        index=4,
        description="Left wing landmark.",
    ),
    CBASKeypoint(
        name="right_wing",
        index=5,
        description="Right wing landmark.",
    ),
    CBASKeypoint(
        name="tail",
        index=6,
        description="Base or center of the tail.",
    ),
    CBASKeypoint(
        name="left_hip",
        index=7,
        description="Left hip landmark.",
    ),
    CBASKeypoint(
        name="right_hip",
        index=8,
        description="Right hip landmark.",
    ),
    CBASKeypoint(
        name="left_knee",
        index=9,
        description="Left knee landmark.",
    ),
    CBASKeypoint(
        name="right_knee",
        index=10,
        description="Right knee landmark.",
    ),
    CBASKeypoint(
        name="left_foot",
        index=11,
        description="Left foot landmark.",
    ),
    CBASKeypoint(
        name="right_foot",
        index=12,
        description="Right foot landmark.",
    ),
)


# =========================================================
# Skeleton Connections
# =========================================================

CBAS_SKELETON_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),   # beak -> head
    (1, 2),   # head -> neck
    (2, 3),   # neck -> body
    (3, 4),   # body -> left wing
    (3, 5),   # body -> right wing
    (3, 6),   # body -> tail
    (3, 7),   # body -> left hip
    (3, 8),   # body -> right hip
    (7, 9),   # left hip -> left knee
    (8, 10),  # right hip -> right knee
    (9, 11),  # left knee -> left foot
    (10, 12), # right knee -> right foot
)


# =========================================================
# Annotation Coordinate System
# =========================================================

class CoordinateSystem(str, Enum):
    """Coordinate systems supported by CBAS."""

    PIXEL = "pixel"

    NORMALIZED = "normalized"

    WORLD = "world"


# =========================================================
# Visibility
# =========================================================

class KeypointVisibility(str, Enum):
    """Visibility state of an anatomical keypoint."""

    VISIBLE = "visible"

    OCCLUDED = "occluded"

    OUT_OF_FRAME = "out_of_frame"

    UNKNOWN = "unknown"


# =========================================================
# Annotation Quality
# =========================================================

class AnnotationQuality(str, Enum):
    """Annotation quality levels."""

    HIGH = "high"

    MEDIUM = "medium"

    LOW = "low"

    INVALID = "invalid"


# =========================================================
# CBAS Metadata
# =========================================================

@dataclass(frozen=True, slots=True)
class CBASMetadata:
    """
    Metadata required for a CBAS-compliant annotation.
    """

    cbas_version: str

    annotation_type: AnnotationType

    annotator: str | None = None

    annotation_quality: AnnotationQuality | None = None

    coordinate_system: CoordinateSystem = (
        CoordinateSystem.PIXEL
    )


# =========================================================
# Helper Functions
# =========================================================

def get_keypoint_by_name(
    name: str,
) -> CBASKeypoint | None:
    """
    Return a canonical CBAS keypoint by name.
    """

    for keypoint in CBAS_KEYPOINTS:
        if keypoint.name == name:
            return keypoint

    return None


def get_keypoint_by_index(
    index: int,
) -> CBASKeypoint | None:
    """
    Return a canonical CBAS keypoint by index.
    """

    for keypoint in CBAS_KEYPOINTS:
        if keypoint.index == index:
            return keypoint

    return None


def get_skeleton_keypoint_count() -> int:
    """
    Return the number of canonical skeleton keypoints.
    """

    return len(CBAS_KEYPOINTS)


def get_skeleton_connection_count() -> int:
    """
    Return the number of canonical skeleton connections.
    """

    return len(CBAS_SKELETON_CONNECTIONS)

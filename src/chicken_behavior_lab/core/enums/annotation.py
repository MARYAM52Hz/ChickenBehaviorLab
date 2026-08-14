"""
ChickenBehaviorLab Annotation Enums
===================================
"""

from __future__ import annotations

from enum import Enum


class AnnotationType(str, Enum):
    """Types of annotations supported by the framework."""

    DETECTION = "detection"

    POSE = "pose"

    SKELETON = "skeleton"

    BEHAVIOR = "behavior"

    EVENT = "event"


class CoordinateSystem(str, Enum):
    """Coordinate systems used by annotations."""

    PIXEL = "pixel"

    NORMALIZED = "normalized"

    WORLD = "world"


class KeypointVisibility(str, Enum):
    """Visibility states for anatomical landmarks."""

    VISIBLE = "visible"

    OCCLUDED = "occluded"

    OUT_OF_FRAME = "out_of_frame"

    UNKNOWN = "unknown"


class AnnotationQuality(str, Enum):
    """Annotation quality levels."""

    HIGH = "high"

    MEDIUM = "medium"

    LOW = "low"

    INVALID = "invalid"

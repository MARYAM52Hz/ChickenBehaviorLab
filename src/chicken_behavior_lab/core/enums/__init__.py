"""
ChickenBehaviorLab Core Enums
=============================
"""

from chicken_behavior_lab.core.enums.behavior import (
    BehaviorCategory,
    BehaviorType,
)

from chicken_behavior_lab.core.enums.event import (
    EventType,
)

from chicken_behavior_lab.core.enums.prediction import (
    RiskLevel,
)

from chicken_behavior_lab.core.enums.keypoints import (
    KeypointType,
)

from chicken_behavior_lab.core.enums.annotation import (
    AnnotationType,
    AnnotationQuality,
    CoordinateSystem,
    KeypointVisibility,
)


__all__ = [
    "BehaviorCategory",
    "BehaviorType",
    "EventType",
    "RiskLevel",
    "KeypointType",
    "AnnotationType",
    "AnnotationQuality",
    "CoordinateSystem",
    "KeypointVisibility",
]

"""
ChickenBehaviorLab Annotation Module
====================================
"""

from chicken_behavior_lab.annotations.labels import (
    BehaviorLabel,
    BehaviorLabelEncoder,
    DEFAULT_BEHAVIOR_LABELS,
    DEFAULT_BEHAVIOR_ENCODER,
)

from chicken_behavior_lab.annotations.schema import (
    BehaviorAnnotation,
    AnnotationSet,
)

from chicken_behavior_lab.annotations.loader import (
    AnnotationLoader,
)


__all__ = [
    "BehaviorLabel",
    "BehaviorLabelEncoder",
    "DEFAULT_BEHAVIOR_LABELS",
    "DEFAULT_BEHAVIOR_ENCODER",
    "BehaviorAnnotation",
    "AnnotationSet",
    "AnnotationLoader",
]

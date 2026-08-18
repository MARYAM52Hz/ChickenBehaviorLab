"""
ChickenBehaviorLab Detection Modules
====================================
"""

from chicken_behavior_lab.detectors.base import (
    BaseDetector,
    BasePoseDetector,
)

from chicken_behavior_lab.detectors.yolo_pose import (
    YOLOPoseDetector,
)

__all__ = [
    "BaseDetector",
    "BasePoseDetector",
    "YOLOPoseDetector",
]

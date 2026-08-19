"""
ChickenBehaviorLab Adapters
===========================
"""

from chicken_behavior_lab.adapters.keypoint_mapping import (
    KeypointMapping,
    YOLO_POSE_KEYPOINT_MAPPING,
    get_cbas_keypoint,
    get_model_index,
)

from chicken_behavior_lab.adapters.yolo_adapter import (
    YOLOResultAdapter,
)

from chicken_behavior_lab.adapters.pose_adapter import (
    YOLOPoseAdapter,
)


__all__ = [
    "KeypointMapping",
    "YOLO_POSE_KEYPOINT_MAPPING",
    "get_cbas_keypoint",
    "get_model_index",
    "YOLOResultAdapter",
    "YOLOPoseAdapter",
]

"""
ChickenBehaviorLab Pose Adapter
================================

Converts raw YOLO-Pose keypoint outputs into
ChickenBehaviorLab pose representations.
"""

from __future__ import annotations

from typing import Any

from chicken_behavior_lab.adapters.keypoint_mapping import (
    get_cbas_keypoint,
)


class YOLOPoseAdapter:
    """
    Adapter for converting YOLO-Pose keypoints into
    the canonical ChickenBehaviorLab representation.
    """

    def adapt_keypoints(
        self,
        keypoints: Any,
    ) -> list[dict]:
        """
        Convert raw YOLO keypoints into CBAS-aware
        keypoint dictionaries.
        """

        adapted: list[dict] = []

        for index, point in enumerate(keypoints):

            cbas_keypoint = get_cbas_keypoint(index)

            if cbas_keypoint is None:
                continue

            adapted.append(
                {
                    "model_index": index,
                    "keypoint": cbas_keypoint.value,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "confidence": (
                        float(point[2])
                        if len(point) > 2
                        else None
                    ),
                }
            )

        return adapted

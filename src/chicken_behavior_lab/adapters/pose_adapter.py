"""
ChickenBehaviorLab Pose Adapter
================================

Converts raw Ultralytics YOLO-Pose outputs into
ChickenBehaviorLab Pose and Keypoint models.
"""

from __future__ import annotations

from typing import Any

from chicken_behavior_lab.adapters.keypoint_mapping import (
    get_cbas_keypoint,
)

from chicken_behavior_lab.core.enums.annotation import (
    KeypointVisibility,
)

from chicken_behavior_lab.models.keypoint import (
    Keypoint,
)

from chicken_behavior_lab.models.pose import (
    Pose,
)


class YOLOPoseAdapter:
    """
    Adapter between Ultralytics YOLO-Pose outputs
    and ChickenBehaviorLab internal pose models.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.25,
    ) -> None:

        self.confidence_threshold = (
            confidence_threshold
        )

    # =====================================================
    # Single Detection
    # =====================================================

    def adapt_detection(
        self,
        keypoints_xy: Any,
        keypoints_conf: Any | None,
        frame_id: str,
        detection_index: int,
        track_id: str | None = None,
    ) -> Pose:
        """
        Convert one detected chicken pose into
        a ChickenBehaviorLab Pose object.
        """

        adapted_keypoints: list[Keypoint] = []

        number_of_keypoints = len(keypoints_xy)

        for model_index in range(
            number_of_keypoints
        ):

            cbas_keypoint = get_cbas_keypoint(
                model_index
            )

            if cbas_keypoint is None:
                continue

            point = keypoints_xy[model_index]

            x = float(point[0])

            y = float(point[1])

            if keypoints_conf is not None:

                confidence = float(
                    keypoints_conf[model_index]
                )

            else:

                confidence = 1.0

            if confidence >= self.confidence_threshold:

                visibility = (
                    KeypointVisibility.VISIBLE
                )

            else:

                visibility = (
                    KeypointVisibility.UNKNOWN
                )

            adapted_keypoints.append(
                Keypoint(
                    keypoint_type=cbas_keypoint,
                    x=x,
                    y=y,
                    confidence=confidence,
                    visibility=visibility,
                )
            )

        pose_confidence = self._calculate_pose_confidence(
            adapted_keypoints
        )

        return Pose(
            pose_id=(
                f"{frame_id}_pose_{detection_index}"
            ),
            frame_id=frame_id,
            keypoints=adapted_keypoints,
            confidence=pose_confidence,
            track_id=track_id,
        )

    # =====================================================
    # Complete YOLO Result
    # =====================================================

    def adapt_result(
        self,
        result: Any,
        frame_id: str,
    ) -> list[Pose]:
        """
        Convert one Ultralytics Results object
        into a list of Pose objects.
        """

        if result.keypoints is None:
            return []

        keypoints_xy = (
            result.keypoints.xy.cpu().numpy()
        )

        keypoints_conf = None

        if result.keypoints.conf is not None:

            keypoints_conf = (
                result.keypoints.conf
                .cpu()
                .numpy()
            )

        poses: list[Pose] = []

        number_of_detections = len(
            keypoints_xy
        )

        for detection_index in range(
            number_of_detections
        ):

            detection_xy = (
                keypoints_xy[detection_index]
            )

            detection_conf = None

            if keypoints_conf is not None:

                detection_conf = (
                    keypoints_conf[
                        detection_index
                    ]
                )

            pose = self.adapt_detection(
                keypoints_xy=detection_xy,
                keypoints_conf=detection_conf,
                frame_id=frame_id,
                detection_index=detection_index,
            )

            poses.append(pose)

        return poses

    # =====================================================
    # Pose Confidence
    # =====================================================

    def _calculate_pose_confidence(
        self,
        keypoints: list[Keypoint],
    ) -> float:
        """
        Calculate the mean confidence of valid
        keypoints in a pose.
        """

        if not keypoints:
            return 0.0

        confidence_sum = sum(
            keypoint.confidence
            for keypoint in keypoints
        )

        return confidence_sum / len(
            keypoints
        )

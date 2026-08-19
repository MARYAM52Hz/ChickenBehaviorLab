"""
ChickenBehaviorLab YOLO-Pose Detector
=====================================

YOLO-Pose based detector and pose estimator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chicken_behavior_lab.adapters.pose_adapter import (
    YOLOPoseAdapter,
)

from chicken_behavior_lab.detectors.base import (
    BasePoseDetector,
)

from chicken_behavior_lab.models.frame import (
    Frame,
)


class YOLOPoseDetector(BasePoseDetector):
    """
    YOLO-Pose adapter used by ChickenBehaviorLab.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:

        self.model_path = Path(
            model_path
        )

        self.confidence_threshold = (
            confidence_threshold
        )

        self.device = device

        self.model: Any | None = None

        self.pose_adapter = (
            YOLOPoseAdapter(
                confidence_threshold=(
                    confidence_threshold
                )
            )
        )

    # =====================================================
    # Load
    # =====================================================

    def load(self) -> None:
        """
        Load the YOLO-Pose model.
        """

        from ultralytics import YOLO

        self.model = YOLO(
            str(self.model_path)
        )

    # =====================================================
    # Raw Prediction
    # =====================================================

    def predict(
        self,
        frame: Frame,
    ) -> list[Any]:

        if self.model is None:
            raise RuntimeError(
                "YOLO-Pose model is not loaded. "
                "Call load() before prediction."
            )

        return self.model.predict(
            source=str(frame.image_path),
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

    # =====================================================
    # Detection
    # =====================================================

    def detect(
        self,
        frame: Frame,
    ) -> list[Any]:

        return self.predict(frame)

    # =====================================================
    # Pose
    # =====================================================

    def detect_pose(
        self,
        frame: Frame,
    ) -> list[Any]:

        results = self.predict(frame)

        poses = []

        for result in results:

            poses.extend(
                self.pose_adapter.adapt_result(
                    result=result,
                    frame_id=frame.frame_id,
                )
            )

        return poses

    # =====================================================
    # Unload
    # =====================================================

    def unload(self) -> None:
        """
        Release the loaded YOLO model.
        """

        self.model = None

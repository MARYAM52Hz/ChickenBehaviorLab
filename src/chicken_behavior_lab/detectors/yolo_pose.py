"""
ChickenBehaviorLab YOLO-Pose Detector
=====================================

YOLO-Pose adapter for chicken detection and pose estimation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chicken_behavior_lab.detectors.base import (
    BasePoseDetector,
)

from chicken_behavior_lab.models.frame import Frame


class YOLOPoseDetector(BasePoseDetector):
    """
    YOLO-Pose based chicken detector.

    This class acts as an adapter between the
    Ultralytics YOLO-Pose implementation and
    ChickenBehaviorLab's internal interfaces.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:

        self.model_path = Path(model_path)

        self.confidence_threshold = confidence_threshold

        self.device = device

        self.model: Any | None = None

    # =====================================================
    # Model Loading
    # =====================================================

    def load(self) -> None:
        """
        Load the YOLO-Pose model.
        """

        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))

    # =====================================================
    # Detection
    # =====================================================

    def detect(
        self,
        frame: Frame,
    ) -> list[Any]:

        if self.model is None:
            raise RuntimeError(
                "YOLO-Pose model has not been loaded. "
                "Call load() before detect()."
            )

        return self.model.predict(
            source=str(frame.image_path),
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

    # =====================================================
    # Pose Detection
    # =====================================================

    def detect_pose(
        self,
        frame: Frame,
    ) -> list[Any]:

        return self.detect(frame)

    # =====================================================
    # Unload
    # =====================================================

    def unload(self) -> None:
        """
        Release the loaded model.
        """

        self.model = None

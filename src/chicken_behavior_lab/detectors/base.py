"""
ChickenBehaviorLab Detector Interfaces
=======================================

Abstract interfaces for chicken detection and pose estimation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from chicken_behavior_lab.models.frame import Frame


class BaseDetector(ABC):
    """
    Base interface for all chicken detectors.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Load detector model and required resources.
        """
        raise NotImplementedError

    @abstractmethod
    def detect(self, frame: Frame) -> list[Any]:
        """
        Detect chickens in a frame.
        """
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        """
        Release model resources.
        """
        raise NotImplementedError
    class BasePoseDetector(BaseDetector):
    """
    Base interface for pose-aware detectors.
    """

    @abstractmethod
    def detect_pose(self, frame: Frame) -> list[Any]:
        """
        Detect chickens and estimate their pose.
        """
        raise NotImplementedError

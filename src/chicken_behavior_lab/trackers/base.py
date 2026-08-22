"""
ChickenBehaviorLab Tracker Interfaces
======================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from chicken_behavior_lab.models.detection import (
    Detection,
)

from chicken_behavior_lab.models.track import (
    Track,
)


class BaseTracker(ABC):
    """
    Abstract interface for chicken trackers.
    """

    @abstractmethod
    def update(
        self,
        detections: list[Detection],
    ) -> list[Track]:
        """
        Associate detections with existing tracks.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Reset tracker state.
        """
        raise NotImplementedError

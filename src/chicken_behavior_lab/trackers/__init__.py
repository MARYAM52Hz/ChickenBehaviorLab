"""
ChickenBehaviorLab Tracking Modules
===================================
"""

from chicken_behavior_lab.trackers.base import (
    BaseTracker,
)

from chicken_behavior_lab.trackers.simple_tracker import (
    SimpleIoUTracker,
)


__all__ = [
    "BaseTracker",
    "SimpleIoUTracker",
]

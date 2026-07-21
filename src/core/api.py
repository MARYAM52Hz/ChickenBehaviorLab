"""
Public API definitions for ChickenBehaviorLab.

These classes define the expected interfaces of the framework.
Actual implementations are provided in their corresponding modules.
"""

from abc import ABC, abstractmethod


class Detector(ABC):

    @abstractmethod
    def detect(self, image):
        """Detect chickens in a single image."""
        pass


class Tracker(ABC):

    @abstractmethod
    def track(self, detections):
        """Track detected chickens across frames."""
        pass


class PoseEstimator(ABC):

    @abstractmethod
    def predict(self, tracks):
        """Estimate chicken poses."""
        pass


class SkeletonBuilder(ABC):

    @abstractmethod
    def build(self, poses):
        """Build skeleton representation."""
        pass


class BehaviorRecognizer(ABC):

    @abstractmethod
    def predict(self, skeleton_sequence):
        """Recognize behaviors."""
        pass


class EventGenerator(ABC):

    @abstractmethod
    def generate(self, behaviors):
        """Generate behavior events."""
        pass


class Analytics(ABC):

    @abstractmethod
    def compute(self, events):
        """Compute flock-level analytics."""
        pass


class MortalityPredictor(ABC):

    @abstractmethod
    def predict(self, analytics):
        """Predict mortality risk."""
        pass

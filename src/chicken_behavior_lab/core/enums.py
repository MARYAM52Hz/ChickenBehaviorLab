"""
ChickenBehaviorLab Core Enumerations
====================================

This module defines all shared enumerations used throughout the
ChickenBehaviorLab framework.

Author:
    ChickenBehaviorLab Contributors

License:
    MIT License
"""

from enum import Enum


# ---------------------------------------------------------
# Detection
# ---------------------------------------------------------

class DetectionStatus(str, Enum):
    """Status of an object detection."""

    DETECTED = "detected"
    FILTERED = "filtered"
    MISSED = "missed"
    OCCLUDED = "occluded"


# ---------------------------------------------------------
# Tracking
# ---------------------------------------------------------

class TrackingStatus(str, Enum):
    """Status of a tracked object."""

    NEW = "new"
    ACTIVE = "active"
    LOST = "lost"
    REMOVED = "removed"


# ---------------------------------------------------------
# Camera
# ---------------------------------------------------------

class CameraType(str, Enum):
    """Supported camera types."""

    RGB = "rgb"
    THERMAL = "thermal"
    DEPTH = "depth"
    INFRARED = "infrared"


# ---------------------------------------------------------
# Skeleton
# ---------------------------------------------------------

class SkeletonType(str, Enum):
    """Skeleton representation."""

    TWO_D = "2d"
    THREE_D = "3d"


# ---------------------------------------------------------
# Behavior
# ---------------------------------------------------------

class BehaviorType(str, Enum):
    """
    Canonical chicken behaviors.
    """

    UNKNOWN = "unknown"

    STANDING = "standing"

    WALKING = "walking"

    RUNNING = "running"

    FEEDING = "feeding"

    DRINKING = "drinking"

    RESTING = "resting"

    SITTING = "sitting"

    PRENING = "preening"

    WING_FLAPPING = "wing_flapping"

    DUST_BATHING = "dust_bathing"

    PERCHING = "perching"

    SOCIAL_INTERACTION = "social_interaction"

    AGGRESSION = "aggression"

    CHASING = "chasing"

    AVOIDANCE = "avoidance"

    SICKNESS = "sickness"

    DEAD = "dead"


# ---------------------------------------------------------
# Event
# ---------------------------------------------------------

class EventType(str, Enum):
    """Behavior event categories."""

    START = "start"

    CONTINUE = "continue"

    END = "end"

    TRANSITION = "transition"

    ANOMALY = "anomaly"


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

class RiskLevel(str, Enum):
    """Mortality risk levels."""

    LOW = "low"

    MEDIUM = "medium"

    HIGH = "high"

    CRITICAL = "critical"


# ---------------------------------------------------------
# Data Split
# ---------------------------------------------------------

class DatasetSplit(str, Enum):

    TRAIN = "train"

    VALIDATION = "validation"

    TEST = "test"


# ---------------------------------------------------------
# Annotation
# ---------------------------------------------------------

class AnnotationStatus(str, Enum):

    RAW = "raw"

    VERIFIED = "verified"

    CORRECTED = "corrected"

    APPROVED = "approved"


# ---------------------------------------------------------
# File Format
# ---------------------------------------------------------

class FileFormat(str, Enum):

    IMAGE = "image"

    VIDEO = "video"

    JSON = "json"

    CSV = "csv"

    YAML = "yaml"

    PARQUET = "parquet"


# ---------------------------------------------------------
# Confidence
# ---------------------------------------------------------

class ConfidenceLevel(str, Enum):

    LOW = "low"

    MEDIUM = "medium"

    HIGH = "high"

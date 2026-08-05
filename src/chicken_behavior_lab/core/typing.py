"""
ChickenBehaviorLab Type Definitions
===================================

Common type aliases used across the framework.

Author:
    ChickenBehaviorLab Contributors

License:
    MIT License
"""

from __future__ import annotations

from pathlib import Path
from typing import NewType
from uuid import UUID


# ---------------------------------------------------------
# Identifiers
# ---------------------------------------------------------

FrameID = NewType("FrameID", str)

VideoID = NewType("VideoID", str)

DetectionID = NewType("DetectionID", str)

TrackID = NewType("TrackID", str)

PoseID = NewType("PoseID", str)

SkeletonID = NewType("SkeletonID", str)

BehaviorID = NewType("BehaviorID", str)

EventID = NewType("EventID", str)

ExperimentID = NewType("ExperimentID", str)

FarmID = NewType("FarmID", str)

CameraID = NewType("CameraID", str)


# ---------------------------------------------------------
# Numeric Types
# ---------------------------------------------------------

Confidence = NewType("Confidence", float)

Probability = NewType("Probability", float)

Timestamp = NewType("Timestamp", float)

FrameNumber = NewType("FrameNumber", int)

FPS = NewType("FPS", float)

Duration = NewType("Duration", float)

Angle = NewType("Angle", float)

Distance = NewType("Distance", float)

Pixel = NewType("Pixel", int)


# ---------------------------------------------------------
# Dataset Types
# ---------------------------------------------------------

Label = NewType("Label", str)

ClassName = NewType("ClassName", str)

FilePath = NewType("FilePath", str)

DirectoryPath = NewType("DirectoryPath", str)


# ---------------------------------------------------------
# Geometry
# ---------------------------------------------------------

Coordinate2D = tuple[float, float]

Coordinate3D = tuple[float, float, float]

BoundingBox = tuple[
    float,
    float,
    float,
    float,
]


# ---------------------------------------------------------
# Generic Types
# ---------------------------------------------------------

UUIDType = UUID

PathLike = Path | str

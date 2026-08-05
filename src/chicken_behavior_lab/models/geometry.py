"""
ChickenBehaviorLab Geometry Models
==================================

Fundamental geometric data structures used throughout the framework.

Author:
    ChickenBehaviorLab Contributors

License:
    MIT License
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# =========================================================
# Point2D
# =========================================================

@dataclass(slots=True)
class Point2D:
    """
    Two-dimensional point.
    """

    x: float
    y: float


# =========================================================
# Point3D
# =========================================================

@dataclass(slots=True)
class Point3D:
    """
    Three-dimensional point.
    """

    x: float
    y: float
    z: float


# =========================================================
# ImageSize
# =========================================================

@dataclass(slots=True)
class ImageSize:
    """
    Image resolution.
    """

    width: int
    height: int


# =========================================================
# BoundingBox
# =========================================================

@dataclass(slots=True)
class BoundingBox:
    """
    Axis-aligned bounding box.
    """

    x_min: float
    y_min: float

    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point2D:
        return Point2D(
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
        )


# =========================================================
# Polygon
# =========================================================

@dataclass(slots=True)
class Polygon:
    """
    Polygon represented by vertices.
    """

    vertices: list[Point2D]


# =========================================================
# Line
# =========================================================

@dataclass(slots=True)
class Line:
    """
    Line segment.
    """

    start: Point2D
    end: Point2D


# =========================================================
# Circle
# =========================================================

@dataclass(slots=True)
class Circle:
    """
    Circle representation.
    """

    center: Point2D
    radius: float

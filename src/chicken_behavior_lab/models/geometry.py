"""
ChickenBehaviorLab Geometry Models
===================================
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """
    Axis-aligned bounding box in image coordinates.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(
            0.0,
            self.x_max - self.x_min,
        )

    @property
    def height(self) -> float:
        return max(
            0.0,
            self.y_max - self.y_min,
        )

    @property
    def center_x(self) -> float:
        return (
            self.x_min + self.x_max
        ) / 2.0

    @property
    def center_y(self) -> float:
        return (
            self.y_min + self.y_max
        ) / 2.0

    @property
    def area(self) -> float:
        return (
            self.width * self.height
        )

    def iou(
        self,
        other: "BoundingBox",
    ) -> float:
        """
        Compute Intersection over Union.
        """

        intersection_x_min = max(
            self.x_min,
            other.x_min,
        )

        intersection_y_min = max(
            self.y_min,
            other.y_min,
        )

        intersection_x_max = min(
            self.x_max,
            other.x_max,
        )

        intersection_y_max = min(
            self.y_max,
            other.y_max,
        )

        intersection_width = max(
            0.0,
            intersection_x_max
            - intersection_x_min,
        )

        intersection_height = max(
            0.0,
            intersection_y_max
            - intersection_y_min,
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        union_area = (
            self.area
            + other.area
            - intersection_area
        )

        if union_area <= 0.0:
            return 0.0

        return (
            intersection_area
            / union_area
        )

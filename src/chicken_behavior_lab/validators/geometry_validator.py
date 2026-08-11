"""
Geometry Validators
===================

Validation utilities for geometric data structures.
"""

from __future__ import annotations

from chicken_behavior_lab.core.exceptions import ValidationError
from chicken_behavior_lab.models.geometry import (
    BoundingBox,
    ImageSize,
    Point2D,
    Point3D,
)
from chicken_behavior_lab.validators.base import BaseValidator


class Point2DValidator(BaseValidator[Point2D]):
    """Validate 2D points."""

    def validate(self, data: Point2D) -> bool:
        return (
            data.x == data.x
            and data.y == data.y
        )

    def validate_or_raise(self, data: Point2D) -> None:
        if not self.validate(data):
            raise ValidationError(
                "Invalid Point2D: coordinates must be finite."
            )


class Point3DValidator(BaseValidator[Point3D]):
    """Validate 3D points."""

    def validate(self, data: Point3D) -> bool:
        return (
            data.x == data.x
            and data.y == data.y
            and data.z == data.z
        )

    def validate_or_raise(self, data: Point3D) -> None:
        if not self.validate(data):
            raise ValidationError(
                "Invalid Point3D: coordinates must be finite."
            )


class ImageSizeValidator(BaseValidator[ImageSize]):
    """Validate image dimensions."""

    def validate(self, data: ImageSize) -> bool:
        return (
            data.width > 0
            and data.height > 0
        )

    def validate_or_raise(self, data: ImageSize) -> None:
        if not self.validate(data):
            raise ValidationError(
                "Image width and height must be positive."
            )


class BoundingBoxValidator(BaseValidator[BoundingBox]):
    """Validate bounding boxes."""

    def validate(self, data: BoundingBox) -> bool:
        return (
            data.x_max >= data.x_min
            and data.y_max >= data.y_min
        )

    def validate_or_raise(self, data: BoundingBox) -> None:
        if not self.validate(data):
            raise ValidationError(
                "Invalid BoundingBox coordinates."
            )

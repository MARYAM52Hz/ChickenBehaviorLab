"""Adapt raw pose keypoints to the canonical project schema."""

from __future__ import annotations

from collections.abc import Mapping

from .contracts import (
    ContractValidationError,
    Keypoint,
    PoseEstimate,
)
from .keypoint_schema import KeypointSchema


class PoseAdapter:
    """Validate and normalize pose keypoints into a fixed node order."""

    def __init__(
        self,
        schema: KeypointSchema,
        *,
        confidence_threshold: float = 0.0,
        missing_policy: str = "raise",
        reject_extra_keypoints: bool = True,
    ) -> None:
        if not isinstance(schema, KeypointSchema):
            raise ContractValidationError(
                "schema must be a KeypointSchema instance."
            )

        if isinstance(confidence_threshold, bool) or not isinstance(
            confidence_threshold, (int, float)
        ):
            raise ContractValidationError(
                "confidence_threshold must be numeric."
            )

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ContractValidationError(
                "confidence_threshold must be between 0 and 1."
            )

        if missing_policy not in {"raise", "mask"}:
            raise ContractValidationError(
                "missing_policy must be 'raise' or 'mask'."
            )

        if not isinstance(reject_extra_keypoints, bool):
            raise ContractValidationError(
                "reject_extra_keypoints must be a boolean."
            )

        self.schema = schema
        self.confidence_threshold = float(confidence_threshold)
        self.missing_policy = missing_policy
        self.reject_extra_keypoints = reject_extra_keypoints

    def adapt(
        self,
        *,
        track_id: int,
        frame_index: int,
        keypoints: Mapping[str, Keypoint | None],
    ) -> PoseEstimate:
        """Normalize keypoints and preserve canonical schema order."""

        if not isinstance(keypoints, Mapping):
            raise ContractValidationError(
                "keypoints must be a mapping of names to Keypoint values."
            )

        if self.reject_extra_keypoints:
            extra_names = set(keypoints) - set(self.schema.names)

            if extra_names:
                raise ContractValidationError(
                    "Unexpected keypoints: "
                    f"{sorted(map(str, extra_names))}."
                )

        missing_names = [
            name
            for name in self.schema.names
            if name not in keypoints or keypoints[name] is None
        ]

        if missing_names and self.missing_policy == "raise":
            raise ContractValidationError(
                "Missing keypoints: "
                f"{missing_names}."
            )

        normalized: dict[str, Keypoint] = {}

        for name in self.schema.names:
            point = keypoints.get(name)

            if point is None:
                # These are sentinel coordinates, not measured positions.
                normalized[name] = Keypoint(
                    x=0.0,
                    y=0.0,
                    confidence=0.0,
                    visible=False,
                )
                continue

            if not isinstance(point, Keypoint):
                raise ContractValidationError(
                    f"Keypoint {name!r} must be a Keypoint instance or None."
                )

            is_visible = (
                point.visible
                and point.confidence >= self.confidence_threshold
            )

            normalized[name] = Keypoint(
                x=point.x,
                y=point.y,
                confidence=point.confidence,
                visible=is_visible,
            )

        # Dict insertion order follows schema.names.
        return PoseEstimate(
            track_id=track_id,
            frame_index=frame_index,
            keypoints=normalized,
        )

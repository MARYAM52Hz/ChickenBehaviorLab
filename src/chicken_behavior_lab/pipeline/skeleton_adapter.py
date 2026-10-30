"""Convert validated pose estimates into canonical skeleton frames."""

from __future__ import annotations

from .contracts import (
    ContractValidationError,
    PoseEstimate,
    SkeletonFrame,
)
from .keypoint_schema import KeypointSchema


class PoseToSkeletonAdapter:
    """Build a SkeletonFrame using a canonical keypoint schema."""

    def __init__(self, schema: KeypointSchema) -> None:
        if not isinstance(schema, KeypointSchema):
            raise ContractValidationError(
                "schema must be a KeypointSchema instance."
            )

        self.schema = schema

    def adapt(self, pose: PoseEstimate) -> SkeletonFrame:
        """Convert a pose into a schema-ordered skeleton frame."""

        if not isinstance(pose, PoseEstimate):
            raise ContractValidationError(
                "pose must be a PoseEstimate instance."
            )

        pose_names = set(pose.keypoints)
        schema_names = set(self.schema.names)

        missing_names = schema_names - pose_names
        extra_names = pose_names - schema_names

        if missing_names:
            raise ContractValidationError(
                "Pose is missing schema keypoints: "
                f"{sorted(missing_names)}."
            )

        if extra_names:
            raise ContractValidationError(
                "Pose contains keypoints outside the schema: "
                f"{sorted(extra_names)}."
            )

        # Rebuild the mapping in canonical order.
        nodes = {
            name: pose.keypoints[name]
            for name in self.schema.names
        }

        return SkeletonFrame(
            track_id=pose.track_id,
            frame_index=pose.frame_index,
            nodes=nodes,
            edges=self.schema.edges,
        )

"""Pipeline orchestration interfaces."""

from .keypoint_schema import KeypointSchema
from .pose_adapter import PoseAdapter
from .skeleton_adapter import PoseToSkeletonAdapter
from .contracts import (
    ContractValidationError,
    Detection,
    Keypoint,
    PoseEstimate,
    SkeletonFrame,
    TrackObservation,
    validate_pose_matches_track,
    validate_skeleton_matches_pose,

)

__all__ = [
    "ChickenBehaviorPipeline",
    "PipelineConfigurationError",
    "PipelineContext",
    "PipelineResult",
    "PipelineStage",
    "PipelineStageError",
    "ContractValidationError",
    "Detection",
    "Keypoint",
    "PoseEstimate",
    "SkeletonFrame",
    "TrackObservation",
    "validate_pose_matches_track",
    "validate_skeleton_matches_pose",
    "KeypointSchema",
    "PoseAdapter",
]

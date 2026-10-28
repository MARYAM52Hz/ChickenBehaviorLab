"""Pipeline orchestration interfaces."""

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
]

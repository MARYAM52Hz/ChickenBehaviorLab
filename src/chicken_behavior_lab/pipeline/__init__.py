
"""Pipeline orchestration and data-contract interfaces."""

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
from .keypoint_schema import KeypointSchema
from .pose_adapter import PoseAdapter
from .skeleton_adapter import PoseToSkeletonAdapter
from .skeleton_graph_data import (
    SkeletonGraphData,
    SkeletonGraphEncoder,
)
from .orchestrator import (
    ChickenBehaviorPipeline,
    PipelineConfigurationError,
    PipelineContext,
    PipelineResult,
    PipelineStage,
    PipelineStageError,
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

    "PoseToSkeletonAdapter",
    "SkeletonGraphData",
    "SkeletonGraphEncoder",
]

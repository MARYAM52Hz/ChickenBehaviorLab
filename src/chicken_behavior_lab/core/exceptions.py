class ChickenBehaviorError(Exception):
    """Base exception for the framework."""


class ValidationError(ChickenBehaviorError):
    """Raised when data validation fails."""


class PipelineError(ChickenBehaviorError):
    """Raised when pipeline execution fails."""


class DetectionError(ChickenBehaviorError):
    """Raised during object detection."""


class TrackingError(ChickenBehaviorError):
    """Raised during tracking."""


class PoseEstimationError(ChickenBehaviorError):
    """Raised during pose estimation."""


class BehaviorRecognitionError(ChickenBehaviorError):
    """Raised during behavior recognition."""


class PredictionError(ChickenBehaviorError):
    """Raised during mortality prediction."""

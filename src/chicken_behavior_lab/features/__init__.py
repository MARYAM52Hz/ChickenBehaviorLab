"""
ChickenBehaviorLab Feature Extraction
======================================

Feature extraction, motion representation,
temporal sequencing, and normalization utilities.
"""
"""
ChickenBehaviorLab Features Module
===================================
"""

from chicken_behavior_lab.features.temporal_features import (
    TemporalFeatureSequence,
)

from chicken_behavior_lab.features.feature_store import (
    FeatureStore,
)

from chicken_behavior_lab.features.position import (
    PositionFeatures,
    PositionFeatureExtractor,
)

from chicken_behavior_lab.features.velocity import (
    VelocityFeatures,
    VelocityFeatureExtractor,
)

from chicken_behavior_lab.features.acceleration import (
    AccelerationFeatures,
    AccelerationFeatureExtractor,
)

from chicken_behavior_lab.features.angles import (
    JointAngleDefinition,
    JointAngleFeatures,
    JointAngleFeatureExtractor,
)

from chicken_behavior_lab.features.motion import (
    MotionFeatureFrame,
    MotionFeatureExtractor,
)

from chicken_behavior_lab.features.temporal_features import (
    TemporalFeatureSequence,
    TemporalFeatureBuilder,
)

from chicken_behavior_lab.features.normalization import (
    NormalizationStatistics,
    MotionFeatureNormalizer,
)


__all__ = [
    # Position
    "PositionFeatures",
    "PositionFeatureExtractor",

    # Velocity
    "VelocityFeatures",
    "VelocityFeatureExtractor",

    # Acceleration
    "AccelerationFeatures",
    "AccelerationFeatureExtractor",

    # Joint angles
    "JointAngleDefinition",
    "JointAngleFeatures",
    "JointAngleFeatureExtractor",

    # Motion representation
    "MotionFeatureFrame",
    "MotionFeatureExtractor",

    # Temporal representation
    "TemporalFeatureSequence",
    "TemporalFeatureBuilder",

    # Normalization
    "NormalizationStatistics",
    "MotionFeatureNormalizer",

    
    "TemporalFeatureSequence",
    "FeatureStore",
]

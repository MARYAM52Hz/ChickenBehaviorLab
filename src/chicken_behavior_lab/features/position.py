"""
ChickenBehaviorLab Position Features
=====================================

Utilities for extracting spatial position features
from tracked chicken skeletons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chicken_behavior_lab.models.tracked_skeleton import (
    TrackedSkeleton,
)


@dataclass(slots=True)
class PositionFeatures:
    """
    Spatial position features for one skeleton.

    Attributes
    ----------
    coordinates:
        Keypoint coordinates with shape:

            (num_keypoints, 2)

        where the last dimension represents:

            [x, y]

    confidence:
        Keypoint confidence values with shape:

            (num_keypoints,)
    """

    coordinates: np.ndarray

    confidence: np.ndarray

    @property
    def num_keypoints(self) -> int:
        """
        Return the number of keypoints.
        """

        return self.coordinates.shape[0]

    @property
    def feature_dimension(self) -> int:
        """
        Return the number of spatial features
        per keypoint.
        """

        return self.coordinates.shape[1]


class PositionFeatureExtractor:
    """
    Extract spatial position features from
    a TrackedSkeleton.
    """

    def extract(
        self,
        observation: TrackedSkeleton,
    ) -> PositionFeatures:
        """
        Extract x/y coordinates and confidence
        from a tracked skeleton.

        Parameters
        ----------
        observation:
            Tracked skeleton for one chicken
            in one frame.

        Returns
        -------
        PositionFeatures
            Spatial keypoint features.
        """

        keypoints = observation.skeleton.keypoints

        if not keypoints:
            return PositionFeatures(
                coordinates=np.empty(
                    (0, 2),
                    dtype=np.float32,
                ),
                confidence=np.empty(
                    (0,),
                    dtype=np.float32,
                ),
            )

        coordinates = np.array(
            [
                [
                    keypoint.x,
                    keypoint.y,
                ]
                for keypoint in keypoints
            ],
            dtype=np.float32,
        )

        confidence = np.array(
            [
                keypoint.confidence
                for keypoint in keypoints
            ],
            dtype=np.float32,
        )

        return PositionFeatures(
            coordinates=coordinates,
            confidence=confidence,
        )

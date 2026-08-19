"""
ChickenBehaviorLab Keypoint Mapping
===================================

Maps model-specific pose keypoints to the canonical
CBAS anatomical keypoint vocabulary.

This module provides a stable interface between:

    Pose Model
        ↓
    Model Keypoint Index
        ↓
    CBAS Keypoint
        ↓
    Skeleton
"""

from __future__ import annotations

from dataclasses import dataclass

from chicken_behavior_lab.core.enums.keypoints import (
    KeypointType,
)


@dataclass(frozen=True, slots=True)
class KeypointMapping:
    """
    Mapping between a model keypoint and a CBAS keypoint.
    """

    model_index: int

    cbas_keypoint: KeypointType

    name: str


# =========================================================
# YOLO-Pose → CBAS Mapping
# =========================================================

YOLO_POSE_KEYPOINT_MAPPING: tuple[
    KeypointMapping, ...
] = (

    KeypointMapping(
        model_index=0,
        cbas_keypoint=KeypointType.BEAK,
        name="beak",
    ),

    KeypointMapping(
        model_index=1,
        cbas_keypoint=KeypointType.HEAD,
        name="head",
    ),

    KeypointMapping(
        model_index=2,
        cbas_keypoint=KeypointType.NECK,
        name="neck",
    ),

    KeypointMapping(
        model_index=3,
        cbas_keypoint=KeypointType.BODY_CENTER,
        name="body_center",
    ),

    KeypointMapping(
        model_index=4,
        cbas_keypoint=KeypointType.LEFT_WING,
        name="left_wing",
    ),

    KeypointMapping(
        model_index=5,
        cbas_keypoint=KeypointType.RIGHT_WING,
        name="right_wing",
    ),

    KeypointMapping(
        model_index=6,
        cbas_keypoint=KeypointType.TAIL,
        name="tail",
    ),

    KeypointMapping(
        model_index=7,
        cbas_keypoint=KeypointType.LEFT_HIP,
        name="left_hip",
    ),

    KeypointMapping(
        model_index=8,
        cbas_keypoint=KeypointType.RIGHT_HIP,
        name="right_hip",
    ),

    KeypointMapping(
        model_index=9,
        cbas_keypoint=KeypointType.LEFT_KNEE,
        name="left_knee",
    ),

    KeypointMapping(
        model_index=10,
        cbas_keypoint=KeypointType.RIGHT_KNEE,
        name="right_knee",
    ),

    KeypointMapping(
        model_index=11,
        cbas_keypoint=KeypointType.LEFT_FOOT,
        name="left_foot",
    ),

    KeypointMapping(
        model_index=12,
        cbas_keypoint=KeypointType.RIGHT_FOOT,
        name="right_foot",
    ),
)


# =========================================================
# Lookup Functions
# =========================================================

def get_cbas_keypoint(
    model_index: int,
) -> KeypointType | None:
    """
    Convert a model keypoint index into its CBAS keypoint.
    """

    for mapping in YOLO_POSE_KEYPOINT_MAPPING:

        if mapping.model_index == model_index:
            return mapping.cbas_keypoint

    return None


def get_model_index(
    keypoint: KeypointType,
) -> int | None:
    """
    Convert a CBAS keypoint into the corresponding
    model keypoint index.
    """

    for mapping in YOLO_POSE_KEYPOINT_MAPPING:

        if mapping.cbas_keypoint == keypoint:
            return mapping.model_index

    return None


def get_mapping(
    model_index: int,
) -> KeypointMapping | None:
    """
    Return the complete mapping record.
    """

    for mapping in YOLO_POSE_KEYPOINT_MAPPING:

        if mapping.model_index == model_index:
            return mapping

    return None

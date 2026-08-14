"""
ChickenBehaviorLab Data Schemas
================================

Canonical lightweight schemas used for exchanging
data between pipeline components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chicken_behavior_lab.standards.cbas import (
    AnnotationType,
    CBASMetadata,
)


@dataclass(slots=True)
class AnnotationRecord:
    """
    Generic CBAS-compatible annotation record.
    """

    annotation_id: str

    annotation_type: AnnotationType

    frame_id: str

    subject_id: str

    data: dict[str, Any]

    metadata: CBASMetadata


@dataclass(slots=True)
class PredictionRecord:
    """
    Generic model prediction record.
    """

    prediction_id: str

    subject_id: str

    frame_id: str | None

    label: str

    confidence: float

    model_name: str

    model_version: str

"""
ChickenBehaviorLab Metadata
===========================

Standard metadata attached to all major outputs.

Author:
    ChickenBehaviorLab Contributors

License:
    MIT License
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class Metadata:
    """
    Standard metadata shared across the framework.
    """

    framework_version: str

    cbas_version: str

    cbo_version: str

    api_version: str

    schema_version: str

    model_name: str

    model_version: str

    experiment_id: str

    farm_id: str

    camera_id: str

    generated_at: datetime

    author: str | None = None

    notes: str | None = None

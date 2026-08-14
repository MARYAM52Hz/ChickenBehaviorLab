"""
ChickenBehaviorLab Prediction Enums
===================================
"""

from __future__ import annotations

from enum import Enum


class RiskLevel(str, Enum):
    """Mortality or abnormal-behavior risk levels."""

    LOW = "low"

    MEDIUM = "medium"

    HIGH = "high"

    CRITICAL = "critical"

    UNKNOWN = "unknown"

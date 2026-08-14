"""
ChickenBehaviorLab Behavior Enums
==================================
"""

from __future__ import annotations

from enum import Enum


class BehaviorCategory(str, Enum):
    """High-level behavioral categories."""

    LOCOMOTION = "locomotion"
    FEEDING = "feeding"
    RESTING = "resting"
    GROOMING = "grooming"
    SOCIAL = "social"
    AGGRESSIVE = "aggressive"
    ABNORMAL = "abnormal"
    UNKNOWN = "unknown"


class BehaviorType(str, Enum):
    """Canonical behavior identifiers."""

    # Locomotion
    STANDING = "standing"
    WALKING = "walking"
    RUNNING = "running"

    # Feeding
    FEEDING = "feeding"
    DRINKING = "drinking"
    FORAGING = "foraging"

    # Resting
    RESTING = "resting"
    SITTING = "sitting"
    LYING = "lying"

    # Grooming
    PREENING = "preening"
    SCRATCHING = "scratching"

    # Social
    SOCIAL_INTERACTION = "social_interaction"
    FOLLOWING = "following"
    CHASING = "chasing"
    AVOIDANCE = "avoidance"

    # Aggressive
    PECKING = "pecking"
    AGGRESSION = "aggression"
    FIGHTING = "fighting"

    # Abnormal
    IMMOBILITY = "immobility"
    ABNORMAL_MOVEMENT = "abnormal_movement"
    ISOLATION = "isolation"
    SICKNESS_INDICATOR = "sickness_indicator"
    DEAD = "dead"

    UNKNOWN = "unknown"

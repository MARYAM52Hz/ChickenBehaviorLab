"""
Chicken Behavior Ontology (CBO)
================================

Reference implementation of the Chicken Behavior Ontology
used by ChickenBehaviorLab.

CBO provides:

- Behavioral categories
- Canonical behavior identifiers
- Parent-child relationships
- Behavior definitions
- Behavior properties
- Ontology versioning

The ontology is intentionally separated from machine-learning
models so that behavior definitions remain stable while
recognition models can evolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# =========================================================
# CBO Version
# =========================================================

CBO_VERSION = "1.0.0"


# =========================================================
# Behavior Categories
# =========================================================

class BehaviorCategory(str, Enum):
    """
    High-level behavioral categories.
    """

    LOCOMOTION = "locomotion"

    FEEDING = "feeding"

    RESTING = "resting"

    GROOMING = "grooming"

    SOCIAL = "social"

    AGGRESSIVE = "aggressive"

    ABNORMAL = "abnormal"

    UNKNOWN = "unknown"


# =========================================================
# Canonical Behavior Types
# =========================================================

class BehaviorType(str, Enum):
    """
    Canonical CBO behavior identifiers.
    """

    # -----------------------------------------------------
    # Locomotion
    # -----------------------------------------------------

    STANDING = "standing"

    WALKING = "walking"

    RUNNING = "running"

    # -----------------------------------------------------
    # Feeding
    # -----------------------------------------------------

    FEEDING = "feeding"

    DRINKING = "drinking"

    FORAGING = "foraging"

    # -----------------------------------------------------
    # Resting
    # -----------------------------------------------------

    SITTING = "sitting"

    LYING = "lying"

    RESTING = "resting"

    # -----------------------------------------------------
    # Grooming
    # -----------------------------------------------------

    PREENING = "preening"

    SCRATCHING = "scratching"

    # -----------------------------------------------------
    # Social
    # -----------------------------------------------------

    SOCIAL_INTERACTION = "social_interaction"

    FOLLOWING = "following"

    CHASING = "chasing"

    AVOIDANCE = "avoidance"

    # -----------------------------------------------------
    # Aggressive
    # -----------------------------------------------------

    PECKING = "pecking"

    AGGRESSION = "aggression"

    FIGHTING = "fighting"

    # -----------------------------------------------------
    # Abnormal
    # -----------------------------------------------------

    IMMOBILITY = "immobility"

    ABNORMAL_MOVEMENT = "abnormal_movement"

    ISOLATION = "isolation"

    SICKNESS_INDICATOR = "sickness_indicator"

    DEAD = "dead"

    UNKNOWN = "unknown"


# =========================================================
# Behavior Definition
# =========================================================

@dataclass(frozen=True, slots=True)
class BehaviorDefinition:
    """
    Formal definition of one CBO behavior.
    """

    behavior: BehaviorType

    category: BehaviorCategory

    label: str

    description: str

    parent: BehaviorType | None = None

    observable: bool = True

    abnormal: bool = False


# =========================================================
# Canonical Behavior Definitions
# =========================================================

CBO_BEHAVIORS: tuple[BehaviorDefinition, ...] = (

    # -----------------------------------------------------
    # Locomotion
    # -----------------------------------------------------

    BehaviorDefinition(
        behavior=BehaviorType.STANDING,
        category=BehaviorCategory.LOCOMOTION,
        label="Standing",
        description=(
            "Chicken remains upright with limited "
            "whole-body displacement."
        ),
    ),

    BehaviorDefinition(
        behavior=BehaviorType.WALKING,
        category=BehaviorCategory.LOCOMOTION,
        label="Walking",
        description=(
            "Chicken moves through the environment "
            "using coordinated leg motion."
        ),
    ),

    BehaviorDefinition(
        behavior=BehaviorType.RUNNING,
        category=BehaviorCategory.LOCOMOTION,
        label="Running",
        description=(
            "Chicken performs rapid locomotion with "
            "higher displacement and activity."
        ),
        parent=BehaviorType.WALKING,
    ),

    # -----------------------------------------------------
    # Feeding
    # -----------------------------------------------------

    BehaviorDefinition(
        behavior=BehaviorType.FEEDING,
        category=BehaviorCategory.FEEDING,
        label="Feeding",
        description=(
            "Chicken interacts with feed material "
            "for ingestion."
        ),
    ),

    BehaviorDefinition(
        behavior=BehaviorType.DRINKING,
        category=BehaviorCategory.FEEDING,
        label="Drinking",
        description=(
            "Chicken interacts with a drinking source "
            "for water consumption."
        ),
    ),

    BehaviorDefinition(
        behavior=BehaviorType.FORAGING,
        category=BehaviorCategory.FEEDING,
        label="Foraging",
        description=(
            "Chicken searches or explores the environment "
            "for food."
        ),
    ),

    # -----------------------------------------------------
    # Resting
    # -----------------------------------------------------

    BehaviorDefinition(
        behavior=BehaviorType.RESTING,
        category=BehaviorCategory.RESTING,
        label="Resting",
        description=(
            "Chicken exhibits low locomotor activity "
            "without active feeding or social behavior."
        ),
    ),

    BehaviorDefinition(
        behavior=BehaviorType.SITTING,
        category=BehaviorCategory.RESTING,
        label="Sitting",
        description=(
            "Chicken remains in a seated posture."
        ),
        parent=BehaviorType.RESTING,
    ),

    BehaviorDefinition(
        behavior=BehaviorType.LYING,
        category=BehaviorCategory.RESTING,
        label="Lying",
        description=(
            "Chicken remains in a lying posture."
        ),
        parent=BehaviorType.RESTING,
    ),

    # -----------------------------------------------------
    # Grooming
    # -----------------------------------------------------

    BehaviorDefinition(
        behavior=BehaviorType.PREENING,
        category=BehaviorCategory.GROOMING,
        label="Preening",
        description=(
            "Chicken cleans or arranges its feathers "
            "using its beak."
        ),
    ),

    BehaviorDefinition(
        behavior=BehaviorType.SCRATCHING,
        category=BehaviorCategory.GROOMING,
        label="Scratching",
        description=(
            "Chicken performs scratching movements "
            "with its feet."
        ),
    ),

    # -----------------------------------------------------
    # Social
    # -----------------------------------------------------

    BehaviorDefinition(
        behavior=BehaviorType.SOCIAL_INTERACTION,
        category=BehaviorCategory.SOCIAL,
        label="Social Interaction",
        description=(
            "Observable interaction between chickens."
        ),
    ),

    BehaviorDefinition(
        behavior=BehaviorType.FOLLOWING,
        category=BehaviorCategory.SOCIAL,
        label="Following",
        description=(
            "One chicken follows the movement of "
            "another chicken."
        ),
    ),

    BehaviorDefinition(
        behavior=BehaviorType.CHASING,
        category=BehaviorCategory.SOCIAL,
        label="Chasing",
        description=(
            "One chicken actively pursues another."
        ),
    ),

    BehaviorDefinition(
        behavior=BehaviorType.AVOIDANCE,
        category=BehaviorCategory.SOCIAL,
        label="Avoidance",
        description=(
            "Chicken moves away from another chicken "
            "or an external stimulus."
        ),
    ),

    # -----------------------------------------------------
    # Aggression
    # -----------------------------------------------------

    BehaviorDefinition(
        behavior=BehaviorType.PECKING,
        category=BehaviorCategory.AGGRESSIVE,
        label="Pecking",
        description=(
            "Chicken directs a pecking action toward "
            "another chicken or an object."
        ),
    ),

    BehaviorDefinition(
        behavior=BehaviorType.AGGRESSION,
        category=BehaviorCategory.AGGRESSIVE,
        label="Aggression",
        description=(
            "Behavior involving aggressive interaction "
            "toward another chicken."
        ),
    ),

    BehaviorDefinition(
        behavior=BehaviorType.FIGHTING,
        category=BehaviorCategory.AGGRESSIVE,
        label="Fighting",
        description=(
            "Sustained aggressive physical interaction "
            "between chickens."
        ),
        parent=BehaviorType.AGGRESSION,
    ),

    # -----------------------------------------------------
    # Abnormal
    # -----------------------------------------------------

    BehaviorDefinition(
        behavior=BehaviorType.IMMOBILITY,
        category=BehaviorCategory.ABNORMAL,
        label="Abnormal Immobility",
        description=(
            "Unusually prolonged lack of movement "
            "relative to the expected behavioral baseline."
        ),
        abnormal=True,
    ),

    BehaviorDefinition(
        behavior=BehaviorType.ABNORMAL_MOVEMENT,
        category=BehaviorCategory.ABNORMAL,
        label="Abnormal Movement",
        description=(
            "Movement pattern substantially different "
            "from the expected behavioral baseline."
        ),
        abnormal=True,
    ),

    BehaviorDefinition(
        behavior=BehaviorType.ISOLATION,
        category=BehaviorCategory.ABNORMAL,
        label="Isolation",
        description=(
            "Chicken remains spatially separated from "
            "the main flock for an unusual duration."
        ),
        abnormal=True,
    ),

    BehaviorDefinition(
        behavior=BehaviorType.SICKNESS_INDICATOR,
        category=BehaviorCategory.ABNORMAL,
        label="Sickness Indicator",
        description=(
            "Observable behavioral pattern that may "
            "indicate reduced welfare or illness."
        ),
        abnormal=True,
    ),

    BehaviorDefinition(
        behavior=BehaviorType.DEAD,
        category=BehaviorCategory.ABNORMAL,
        label="Dead",
        description=(
            "Chicken exhibits visual characteristics "
            "consistent with mortality."
        ),
        abnormal=True,
    ),
)


# =========================================================
# Ontology Lookup Functions
# =========================================================

def get_behavior_definition(
    behavior: BehaviorType,
) -> BehaviorDefinition | None:
    """
    Retrieve the formal definition of a behavior.
    """

    for definition in CBO_BEHAVIORS:
        if definition.behavior == behavior:
            return definition

    return None


def get_behaviors_by_category(
    category: BehaviorCategory,
) -> tuple[BehaviorDefinition, ...]:
    """
    Return all behaviors belonging to a category.
    """

    return tuple(
        definition
        for definition in CBO_BEHAVIORS
        if definition.category == category
    )


def get_child_behaviors(
    parent: BehaviorType,
) -> tuple[BehaviorDefinition, ...]:
    """
    Return behaviors whose parent is the given behavior.
    """

    return tuple(
        definition
        for definition in CBO_BEHAVIORS
        if definition.parent == parent
    )


def is_abnormal_behavior(
    behavior: BehaviorType,
) -> bool:
    """
    Determine whether a behavior is explicitly classified
    as abnormal in CBO.
    """

    definition = get_behavior_definition(behavior)

    if definition is None:
        return False

    return definition.abnormal

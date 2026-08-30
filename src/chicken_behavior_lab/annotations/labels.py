"""
ChickenBehaviorLab Behavior Labels
===================================

Canonical behavior label definitions and encoding utilities.

This module provides a stable mapping between CBO behavior
identifiers and integer class indices used by machine-learning
models.

Important:
    The numerical class IDs are internal ML identifiers.
    They must not replace the canonical CBO identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BehaviorLabel:
    """
    Definition of a single behavior class.

    Parameters
    ----------
    behavior_id:
        Stable CBO identifier.

    name:
        Human-readable behavior name.

    description:
        Optional description.
    """

    behavior_id: str

    name: str

    description: str = ""


class BehaviorLabelEncoder:
    """
    Encode and decode CBO behavior identifiers.

    The encoder maintains a deterministic mapping:

        CBO behavior ID
                ↓
        integer class index

    Example
    -------
    feeding -> 0
    walking -> 1
    standing -> 2
    """

    def __init__(
        self,
        labels: list[BehaviorLabel],
    ) -> None:

        if not labels:

            raise ValueError(
                "labels cannot be empty."
            )

        self.labels = tuple(labels)

        self._validate_labels()

        self._behavior_to_index = {
            label.behavior_id: index
            for index, label
            in enumerate(self.labels)
        }

        self._index_to_behavior = {
            index: label.behavior_id
            for index, label
            in enumerate(self.labels)
        }

    def _validate_labels(self) -> None:
        """
        Validate label definitions.
        """

        behavior_ids = [
            label.behavior_id
            for label in self.labels
        ]

        if any(
            not behavior_id
            for behavior_id
            in behavior_ids
        ):

            raise ValueError(
                "behavior_id cannot be empty."
            )

        if len(
            behavior_ids
        ) != len(
            set(behavior_ids)
        ):

            raise ValueError(
                "behavior_id values must be unique."
            )

    @property
    def num_classes(self) -> int:
        """
        Return number of behavior classes.
        """

        return len(self.labels)

    def encode(
        self,
        behavior_id: str,
    ) -> int:
        """
        Convert a CBO behavior ID into
        an integer class index.
        """

        if behavior_id not in (
            self._behavior_to_index
        ):

            raise KeyError(
                f"Unknown behavior_id: "
                f"{behavior_id}"
            )

        return self._behavior_to_index[
            behavior_id
        ]

    def decode(
        self,
        index: int,
    ) -> str:
        """
        Convert an integer class index
        back to its CBO behavior ID.
        """

        if index not in (
            self._index_to_behavior
        ):

            raise KeyError(
                f"Unknown class index: "
                f"{index}"
            )

        return self._index_to_behavior[
            index
        ]

    def get_label(
        self,
        behavior_id: str,
    ) -> BehaviorLabel:
        """
        Return the complete BehaviorLabel.
        """

        index = self.encode(
            behavior_id
        )

        return self.labels[index]

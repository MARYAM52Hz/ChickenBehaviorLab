"""Canonical behavior label definitions and integer encodings.

Canonical behavior IDs remain stable identifiers; integer class indices are
internal to machine-learning models.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BehaviorLabel:
    """Definition of one canonical behavior class."""

    behavior_id: str
    name: str
    description: str = ""


class BehaviorLabelEncoder:
    """Deterministic mapping between behavior IDs and ML class indices."""

    def __init__(self, labels: list[BehaviorLabel]) -> None:
        if not labels:
            raise ValueError("labels cannot be empty.")
        if any(not isinstance(label, BehaviorLabel) for label in labels):
            raise TypeError("labels must contain only BehaviorLabel instances.")
        self.labels = tuple(labels)
        self._validate_labels()
        self._behavior_to_index = {
            label.behavior_id: index for index, label in enumerate(self.labels)
        }
        self._index_to_behavior = {
            index: label.behavior_id for index, label in enumerate(self.labels)
        }

    def _validate_labels(self) -> None:
        behavior_ids = [label.behavior_id for label in self.labels]
        if any(not isinstance(value, str) or not value.strip() for value in behavior_ids):
            raise ValueError("behavior_id values must be non-empty strings.")
        if len(behavior_ids) != len(set(behavior_ids)):
            raise ValueError("behavior_id values must be unique.")

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def encode(self, behavior_id: str) -> int:
        try:
            return self._behavior_to_index[behavior_id]
        except KeyError as exc:
            raise KeyError(f"Unknown behavior_id: {behavior_id}") from exc

    def decode(self, index: int) -> str:
        try:
            return self._index_to_behavior[index]
        except KeyError as exc:
            raise KeyError(f"Unknown class index: {index}") from exc

    def get_label(self, behavior_id: str) -> BehaviorLabel:
        return self.labels[self.encode(behavior_id)]


DEFAULT_BEHAVIOR_LABELS = [
    BehaviorLabel(
        behavior_id="feeding",
        name="Feeding",
        description="Chicken performing feeding-related activity.",
    ),
    BehaviorLabel(
        behavior_id="walking",
        name="Walking",
        description="Chicken performing locomotion.",
    ),
    BehaviorLabel(
        behavior_id="standing",
        name="Standing",
        description="Chicken maintaining a standing posture.",
    ),
    BehaviorLabel(
        behavior_id="drinking",
        name="Drinking",
        description="Chicken performing drinking-related activity.",
    ),
    BehaviorLabel(
        behavior_id="resting",
        name="Resting",
        description="Chicken performing resting-related activity.",
    ),
]

DEFAULT_BEHAVIOR_ENCODER = BehaviorLabelEncoder(DEFAULT_BEHAVIOR_LABELS)

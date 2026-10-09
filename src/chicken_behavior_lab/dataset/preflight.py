
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from chicken_behavior_lab.dataset.sample import GraphSample


@dataclass(slots=True)
class ClassCoverageReport:
    """Class distribution and coverage across dataset splits."""

    train_classes: set[str]
    validation_classes: set[str]
    test_classes: set[str]

    validation_only_classes: set[str]
    test_only_classes: set[str]

    train_counts: dict[str, int]
    validation_counts: dict[str, int]
    test_counts: dict[str, int]

    @property
    def is_valid(self) -> bool:
        """Whether all validation/test classes occur in training."""

        return not (
            self.validation_only_classes
            or self.test_only_classes
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation of the report."""

        return {
            "is_valid": self.is_valid,
            "train_classes": sorted(self.train_classes),
            "validation_classes": sorted(self.validation_classes),
            "test_classes": sorted(self.test_classes),
            "validation_only_classes": sorted(
                self.validation_only_classes
            ),
            "test_only_classes": sorted(self.test_only_classes),
            "train_counts": dict(sorted(self.train_counts.items())),
            "validation_counts": dict(
                sorted(self.validation_counts.items())
            ),
            "test_counts": dict(sorted(self.test_counts.items())),
        }


def _count_classes(
    samples: Sequence[GraphSample],
    split_name: str,
) -> Counter[str]:
    """Count behavior labels and validate their basic representation."""

    if not samples:
        raise ValueError(
            f"{split_name} split is empty; class coverage "
            "cannot be evaluated."
        )

    counts: Counter[str] = Counter()

    for sample in samples:
        behavior_id = sample.behavior_id

        if not isinstance(behavior_id, str) or not behavior_id.strip():
            raise ValueError(
                f"{split_name} contains a sample with an invalid "
                "behavior_id."
            )

        counts[behavior_id] += 1

    return counts


def inspect_class_coverage(
    train_samples: Sequence[GraphSample],
    validation_samples: Sequence[GraphSample],
    test_samples: Sequence[GraphSample],
) -> ClassCoverageReport:
    """
    Inspect class coverage without modifying samples or labels.

    The training split defines the known class space. Validation and
    test may contain only classes already represented in training.
    """

    train_counts = _count_classes(
        train_samples,
        "Training",
    )

    validation_counts = _count_classes(
        validation_samples,
        "Validation",
    )

    test_counts = _count_classes(
        test_samples,
        "Test",
    )

    train_classes = set(train_counts)
    validation_classes = set(validation_counts)
    test_classes = set(test_counts)

    return ClassCoverageReport(
        train_classes=train_classes,
        validation_classes=validation_classes,
        test_classes=test_classes,
        validation_only_classes=(
            validation_classes - train_classes
        ),
        test_only_classes=(
            test_classes - train_classes
        ),
        train_counts=dict(train_counts),
        validation_counts=dict(validation_counts),
        test_counts=dict(test_counts),
    )


def validate_class_coverage(
    train_samples: Sequence[GraphSample],
    validation_samples: Sequence[GraphSample],
    test_samples: Sequence[GraphSample],
) -> ClassCoverageReport:
    """
    Validate class coverage before building the label mapping.

    Raises:
        ValueError: If a split is empty or validation/test contains
            a class absent from training.
    """

    report = inspect_class_coverage(
        train_samples=train_samples,
        validation_samples=validation_samples,
        test_samples=test_samples,
    )

    if report.validation_only_classes:
        raise ValueError(
            "Validation split contains behavior classes absent "
            "from training: "
            f"{sorted(report.validation_only_classes)}. "
            "Adjust the group split; do not build the label mapping "
            "from validation or test data."
        )

    if report.test_only_classes:
        raise ValueError(
            "Test split contains behavior classes absent "
            "from training: "
            f"{sorted(report.test_only_classes)}. "
            "Adjust the group split; do not build the label mapping "
            "from validation or test data."
        )

    return report

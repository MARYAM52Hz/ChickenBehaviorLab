from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from chicken_behavior_lab.dataset.sample import GraphSample


@dataclass(slots=True)
class ClassCoverageReport:
    """Class coverage across train/validation/test splits."""

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
        """Return whether validation and test use known train classes."""

        return not (
            self.validation_only_classes
            or self.test_only_classes
        )


def inspect_class_coverage(
    train_samples: Sequence[GraphSample],
    validation_samples: Sequence[GraphSample],
    test_samples: Sequence[GraphSample],
) -> ClassCoverageReport:
    """
    Inspect behavior-class coverage without modifying any samples.

    Training classes are treated as the canonical supervised label space.
    """

    train_counts = Counter(
        sample.behavior_id
        for sample in train_samples
    )

    validation_counts = Counter(
        sample.behavior_id
        for sample in validation_samples
    )

    test_counts = Counter(
        sample.behavior_id
        for sample in test_samples
    )

    train_classes = set(train_counts)
    validation_classes = set(validation_counts)
    test_classes = set(test_counts)

    validation_only_classes = (
        validation_classes - train_classes
    )

    test_only_classes = (
        test_classes - train_classes
    )

    return ClassCoverageReport(
        train_classes=train_classes,
        validation_classes=validation_classes,
        test_classes=test_classes,
        validation_only_classes=validation_only_classes,
        test_only_classes=test_only_classes,
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
    Validate that validation/test classes are represented in training.

    Raises:
        ValueError:
            If validation or test contains a behavior class absent from
            the training split.
    """

    report = inspect_class_coverage(
        train_samples=train_samples,
        validation_samples=validation_samples,
        test_samples=test_samples,
    )

    if report.validation_only_classes:
        classes = sorted(
            report.validation_only_classes
        )

        raise ValueError(
            "Validation split contains behavior classes absent "
            f"from training: {classes}. "
            "The label mapping must be derived from training only."
        )

    if report.test_only_classes:
        classes = sorted(
            report.test_only_classes
        )

        raise ValueError(
            "Test split contains behavior classes absent "
            f"from training: {classes}. "
            "The label mapping must be derived from training only."
        )

    return report

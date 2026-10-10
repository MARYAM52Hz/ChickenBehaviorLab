
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from chicken_behavior_lab.dataset.sample import GraphSample


@dataclass(slots=True)
class SplitDiagnosticsReport:
    """Serializable diagnostics for a group-aware dataset split."""

    group_key: str
    sample_counts: dict[str, int]
    sample_ratios: dict[str, float]
    group_counts: dict[str, int]
    class_counts: dict[str, dict[str, int]]
    group_overlaps: dict[str, list[str]]
    duplicate_sample_ids: list[str]
    train_classes: list[str]
    validation_only_classes: list[str]
    test_only_classes: list[str]

    @property
    def has_group_leakage(self) -> bool:
        """Return whether any grouping identity appears in multiple splits."""
        return any(bool(groups) for groups in self.group_overlaps.values())

    @property
    def has_unknown_classes(self) -> bool:
        """Return whether validation or test contains classes absent from train."""
        return bool(self.validation_only_classes or self.test_only_classes)

    @property
    def is_valid(self) -> bool:
        """Return whether core split-integrity checks pass."""
        return (
            not self.has_group_leakage
            and not self.duplicate_sample_ids
            and not self.has_unknown_classes
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to JSON-compatible Python values."""
        return {
            "is_valid": self.is_valid,
            "group_key": self.group_key,
            "has_group_leakage": self.has_group_leakage,
            "has_unknown_classes": self.has_unknown_classes,
            "sample_counts": dict(self.sample_counts),
            "sample_ratios": dict(self.sample_ratios),
            "group_counts": dict(self.group_counts),
            "class_counts": {
                split: dict(sorted(counts.items()))
                for split, counts in self.class_counts.items()
            },
            "group_overlaps": {
                name: list(groups)
                for name, groups in self.group_overlaps.items()
            },
            "duplicate_sample_ids": list(self.duplicate_sample_ids),
            "train_classes": list(self.train_classes),
            "validation_only_classes": list(self.validation_only_classes),
            "test_only_classes": list(self.test_only_classes),
        }


def _group_id(sample: GraphSample, group_key: str) -> str:
    """Extract the group identity used to detect cross-split leakage."""
    metadata = sample.metadata or {}

    if group_key == "video_id":
        video_id = metadata.get("video_id")
        if video_id is None or not str(video_id).strip():
            raise ValueError(
                f"Sample {sample.sample_id!r} has no valid video_id."
            )
        return str(video_id)

    if group_key == "track_id":
        track_id = metadata.get("track_id")
        if track_id is None or not str(track_id).strip():
            raise ValueError(
                f"Sample {sample.sample_id!r} has no valid track_id."
            )

        # Track identifiers may repeat across videos, so include video_id.
        video_id = metadata.get("video_id")
        if video_id is not None and str(video_id).strip():
            return f"{video_id}::track::{track_id}"

        return str(track_id)

    raise ValueError("group_key must be either 'video_id' or 'track_id'.")


def _validate_samples(
    samples: Sequence[GraphSample],
    split_name: str,
) -> None:
    """Validate basic sample integrity before computing diagnostics."""
    if not samples:
        raise ValueError(f"{split_name} split is empty.")

    for sample in samples:
        if not isinstance(sample.sample_id, str) or not sample.sample_id.strip():
            raise ValueError(
                f"{split_name} contains a sample with an invalid sample_id."
            )
        if not isinstance(sample.behavior_id, str) or not sample.behavior_id.strip():
            raise ValueError(
                f"{split_name} contains a sample with an invalid behavior_id."
            )


def inspect_split_quality(
    train_samples: Sequence[GraphSample],
    validation_samples: Sequence[GraphSample],
    test_samples: Sequence[GraphSample],
    *,
    group_key: str = "video_id",
) -> SplitDiagnosticsReport:
    """Inspect group isolation, class coverage, and split-size distributions.

    This function does not modify samples or rebalance the splits.
    Split samples before generating overlapping temporal windows.
    """
    splits = {
        "train": list(train_samples),
        "validation": list(validation_samples),
        "test": list(test_samples),
    }

    for split_name, samples in splits.items():
        _validate_samples(samples, split_name)

    groups = {
        split_name: {
            _group_id(sample, group_key)
            for sample in samples
        }
        for split_name, samples in splits.items()
    }

    group_overlaps = {
        "train_validation": sorted(groups["train"] & groups["validation"]),
        "train_test": sorted(groups["train"] & groups["test"]),
        "validation_test": sorted(groups["validation"] & groups["test"]),
    }

    sample_id_counts: Counter[str] = Counter(
        sample.sample_id
        for samples in splits.values()
        for sample in samples
    )
    duplicate_sample_ids = sorted(
        sample_id
        for sample_id, count in sample_id_counts.items()
        if count > 1
    )

    sample_counts = {
        split_name: len(samples)
        for split_name, samples in splits.items()
    }
    total_samples = sum(sample_counts.values())

    sample_ratios = {
        split_name: count / total_samples
        for split_name, count in sample_counts.items()
    }

    class_counts = {
        split_name: dict(Counter(
            sample.behavior_id for sample in samples
        ))
        for split_name, samples in splits.items()
    }

    train_classes = set(class_counts["train"])
    validation_classes = set(class_counts["validation"])
    test_classes = set(class_counts["test"])

    return SplitDiagnosticsReport(
        group_key=group_key,
        sample_counts=sample_counts,
        sample_ratios=sample_ratios,
        group_counts={
            split_name: len(split_groups)
            for split_name, split_groups in groups.items()
        },
        class_counts=class_counts,
        group_overlaps=group_overlaps,
        duplicate_sample_ids=duplicate_sample_ids,
        train_classes=sorted(train_classes),
        validation_only_classes=sorted(validation_classes - train_classes),
        test_only_classes=sorted(test_classes - train_classes),
    )

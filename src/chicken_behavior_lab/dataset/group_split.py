from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from chicken_behavior_lab.dataset.sample import GraphSample
from chicken_behavior_lab.core.identifiers import normalize_track_id


GroupKey = Literal["video_id", "track_id"]


def stratified_group_assignment(
    group_labels: dict[str, Sequence[str]],
    ratios: Sequence[float],
    *,
    seed: int = 42,
    n_restarts: int = 64,
) -> dict[str, int]:
    """Assign whole groups to splits while approximately balancing classes."""
    import random
    from collections import Counter

    if not group_labels:
        raise ValueError("Cannot stratify an empty group mapping.")
    if len(ratios) < 2 or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
        for value in ratios
    ):
        raise ValueError("At least two positive split ratios are required.")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0.")
    if len(group_labels) < len(ratios):
        raise ValueError(
            f"At least {len(ratios)} distinct groups are required for "
            f"{len(ratios)}-way splitting."
        )
    if n_restarts < 1:
        raise ValueError("n_restarts must be at least 1.")

    groups = sorted(group_labels)
    labels_by_group = {
        group: tuple(str(label) for label in group_labels[group])
        for group in groups
    }
    if any(not labels for labels in labels_by_group.values()):
        raise ValueError("Every group must contain at least one labeled sample.")

    total_classes = Counter(
        label for labels in labels_by_group.values() for label in labels
    )
    class_group_counts = Counter(
        label for labels in labels_by_group.values() for label in set(labels)
    )
    total_samples = sum(len(labels) for labels in labels_by_group.values())
    n_splits = len(ratios)

    def cost(assignment: dict[str, int]) -> float:
        split_class_counts = [Counter() for _ in ratios]
        split_group_counts = [0 for _ in ratios]
        split_sample_counts = [0 for _ in ratios]
        split_class_groups = [Counter() for _ in ratios]
        for group, split_index in assignment.items():
            labels = labels_by_group[group]
            split_group_counts[split_index] += 1
            split_sample_counts[split_index] += len(labels)
            split_class_counts[split_index].update(labels)
            split_class_groups[split_index].update(set(labels))

        score = 0.0
        for index, ratio in enumerate(ratios):
            if split_group_counts[index] == 0:
                score += 100.0
            group_error = (
                split_group_counts[index] - len(groups) * ratio
            ) / max(len(groups), 1)
            sample_error = (
                split_sample_counts[index] - total_samples * ratio
            ) / max(total_samples, 1)
            score += 0.35 * group_error**2 + 0.65 * sample_error**2

        for label, total_count in total_classes.items():
            for index, ratio in enumerate(ratios):
                target = total_count * ratio
                error = (
                    split_class_counts[index][label] - target
                ) / max(total_count, 1)
                score += 1.5 * error**2

            if split_class_counts[0][label] == 0:
                score += 8.0
            required_coverage = min(n_splits, class_group_counts[label])
            actual_coverage = sum(
                split_class_groups[index][label] > 0
                for index in range(n_splits)
            )
            score += 1.25 * max(0, required_coverage - actual_coverage)

        return score

    best_assignment: dict[str, int] | None = None
    best_cost = float("inf")
    base_rng = random.Random(seed)
    rarity = {
        group: sum(
            1.0 / class_group_counts[label]
            for label in set(labels_by_group[group])
        )
        for group in groups
    }

    for _ in range(n_restarts):
        rng = random.Random(base_rng.randrange(0, 2**32))
        ordered = sorted(
            groups,
            key=lambda group: (
                -rarity[group],
                -len(labels_by_group[group]),
                rng.random(),
            ),
        )
        assignment: dict[str, int] = {}
        for group in ordered:
            candidate_costs = []
            for split_index in range(n_splits):
                trial = dict(assignment)
                trial[group] = split_index
                candidate_costs.append((cost(trial), rng.random(), split_index))
            assignment[group] = min(candidate_costs)[2]

        # Exhaustive single-group relocation is useful for small datasets,
        # but its repeated full-cost evaluation scales cubically. For larger
        # collections, rely on independent greedy restarts instead.
        if len(groups) <= 32:
            improved = True
            while improved:
                improved = False
                current_cost = cost(assignment)
                best_move = None
                best_move_cost = current_cost
                for group in groups:
                    source = assignment[group]
                    if sum(value == source for value in assignment.values()) <= 1:
                        continue
                    for destination in range(n_splits):
                        if destination == source:
                            continue
                        trial = dict(assignment)
                        trial[group] = destination
                        trial_cost = cost(trial)
                        if trial_cost < best_move_cost - 1e-12:
                            best_move = (group, destination)
                            best_move_cost = trial_cost
                if best_move is not None:
                    assignment[best_move[0]] = best_move[1]
                    improved = True

        final_cost = cost(assignment)
        if final_cost < best_cost:
            best_cost = final_cost
            best_assignment = dict(assignment)

    assert best_assignment is not None
    return best_assignment


@dataclass(slots=True)
class GroupSplitResult:
    train: list[GraphSample]
    validation: list[GraphSample]
    test: list[GraphSample]

    train_groups: list[str]
    validation_groups: list[str]
    test_groups: list[str]

    def validate(self) -> None:
        train_set = set(self.train_groups)
        validation_set = set(self.validation_groups)
        test_set = set(self.test_groups)

        if train_set & validation_set:
            raise ValueError(
                "Train and validation groups overlap."
            )

        if train_set & test_set:
            raise ValueError(
                "Train and test groups overlap."
            )

        if validation_set & test_set:
            raise ValueError(
                "Validation and test groups overlap."
            )

        sample_ids: set[str] = set()

        for samples in (
            self.train,
            self.validation,
            self.test,
        ):
            for sample in samples:
                if sample.sample_id in sample_ids:
                    raise ValueError(
                        "A sample appears in multiple splits: "
                        f"{sample.sample_id}"
                    )

                sample_ids.add(sample.sample_id)


class GroupAwareSplitter:
    """
    Split graph samples by group.

    The split is performed before temporal windows are created.

    Supported groups:
        - video_id
        - track_id

    This prevents overlapping temporal windows from the same
    recording/track from appearing in different splits.
    """

    def __init__(
        self,
        *,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        group_key: GroupKey = "video_id",
    ) -> None:

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.group_key = group_key

        self._validate_ratios()

    def _validate_ratios(self) -> None:
        ratios = (
            self.train_ratio,
            self.validation_ratio,
            self.test_ratio,
        )

        if any(ratio <= 0.0 for ratio in ratios):
            raise ValueError(
                "All split ratios must be greater than zero."
            )

        total = sum(ratios)

        if abs(total - 1.0) > 1e-8:
            raise ValueError(
                "train_ratio + validation_ratio + "
                "test_ratio must equal 1.0."
            )

    def split(
        self,
        samples: Sequence[GraphSample],
        *,
        seed: int = 42,
        shuffle_groups: bool = True,
    ) -> GroupSplitResult:

        samples = list(samples)

        if not samples:
            raise ValueError(
                "Cannot split an empty dataset."
            )

        groups = self._collect_groups(
            samples
        )

        if len(groups) < 3:
            raise ValueError(
                "At least three distinct groups are required "
                "for train/validation/test splitting."
            )

        ordered_groups = sorted(groups)
        if shuffle_groups:
            group_labels = {
                group: [sample.behavior_id for sample in group_samples]
                for group, group_samples in groups.items()
            }
            assignment = stratified_group_assignment(
                group_labels,
                (self.train_ratio, self.validation_ratio, self.test_ratio),
                seed=seed,
            )
            train_groups = [
                group for group in ordered_groups if assignment[group] == 0
            ]
            validation_groups = [
                group for group in ordered_groups if assignment[group] == 1
            ]
            test_groups = [
                group for group in ordered_groups if assignment[group] == 2
            ]
        else:
            train_count, validation_count = self._calculate_group_counts(
                len(ordered_groups)
            )
            train_groups = ordered_groups[:train_count]
            validation_groups = ordered_groups[
                train_count:train_count + validation_count
            ]
            test_groups = ordered_groups[train_count + validation_count:]

        train_group_set = set(
            train_groups
        )

        validation_group_set = set(
            validation_groups
        )

        test_group_set = set(
            test_groups
        )

        train_samples: list[GraphSample] = []
        validation_samples: list[GraphSample] = []
        test_samples: list[GraphSample] = []

        for sample in samples:

            group = self._get_group(
                sample
            )

            if group in train_group_set:
                train_samples.append(
                    sample
                )

            elif group in validation_group_set:
                validation_samples.append(
                    sample
                )

            elif group in test_group_set:
                test_samples.append(
                    sample
                )

            else:
                raise RuntimeError(
                    f"Group '{group}' was not assigned "
                    "to any split."
                )

        result = GroupSplitResult(
            train=train_samples,
            validation=validation_samples,
            test=test_samples,
            train_groups=sorted(
                train_group_set
            ),
            validation_groups=sorted(
                validation_group_set
            ),
            test_groups=sorted(
                test_group_set
            ),
        )

        result.validate()

        self._validate_non_empty_splits(
            result
        )

        return result

    def _collect_groups(
        self,
        samples: Sequence[GraphSample],
    ) -> dict[str, list[GraphSample]]:

        groups: dict[
            str,
            list[GraphSample],
        ] = {}

        for sample in samples:

            group = self._get_group(
                sample
            )

            groups.setdefault(
                group,
                [],
            ).append(sample)

        return groups

    def _get_group(
        self,
        sample: GraphSample,
    ) -> str:

        metadata = sample.metadata or {}

        if self.group_key == "video_id":

            video_id = metadata.get(
                "video_id"
            )

            if video_id is None or not str(video_id).strip():
                raise ValueError(
                    "Cannot perform video-level split because "
                    f"sample '{sample.sample_id}' has no "
                    "usable metadata.video_id."
                )

            return str(video_id).strip()

        if self.group_key == "track_id":

            track_id = metadata.get(
                "track_id"
            )

            if track_id is None:
                raise ValueError(
                    "Cannot perform track-level split because "
                    f"sample '{sample.sample_id}' has no "
                    "metadata.track_id."
                )

            video_id = metadata.get(
                "video_id"
            )
            if video_id is None or not str(video_id).strip():
                raise ValueError(
                    "Cannot perform track-level split because "
                    f"sample '{sample.sample_id}' has no "
                    "usable metadata.video_id."
                )

            canonical_video_id = str(video_id).strip()
            canonical_track_id = normalize_track_id(track_id)
            return f"{canonical_video_id}::track::{canonical_track_id}"

        raise ValueError(
            f"Unsupported group_key: {self.group_key}"
        )

    def _calculate_group_counts(
        self,
        number_of_groups: int,
    ) -> tuple[int, int]:

        if number_of_groups < 3:
            raise ValueError(
                "At least three groups are required."
            )

        train_count = int(
            number_of_groups
            * self.train_ratio
        )

        validation_count = int(
            number_of_groups
            * self.validation_ratio
        )

        # Guarantee that every split receives
        # at least one group.
        train_count = max(
            train_count,
            1,
        )

        validation_count = max(
            validation_count,
            1,
        )

        if (
            train_count
            + validation_count
            >= number_of_groups
        ):
            validation_count = (
                number_of_groups
                - train_count
                - 1
            )

        if validation_count < 1:
            train_count = max(
                number_of_groups - 2,
                1,
            )

            validation_count = 1

        return (
            train_count,
            validation_count,
        )

    @staticmethod
    def _validate_non_empty_splits(
        result: GroupSplitResult,
    ) -> None:

        if not result.train:
            raise ValueError(
                "Train split is empty."
            )

        if not result.validation:
            raise ValueError(
                "Validation split is empty."
            )

        if not result.test:
            raise ValueError(
                "Test split is empty."
            )


@dataclass(slots=True, frozen=True)
class SplitGroups:
    """Disjoint train, validation, and test group identifiers."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def validate(self) -> None:
        train = set(self.train)
        validation = set(self.validation)
        test = set(self.test)
        if train & validation or train & test or validation & test:
            raise ValueError("Group splits must be disjoint.")
        if not train or not validation or not test:
            raise ValueError("Train, validation, and test groups must all be non-empty.")


def build_group_id(sample, group_by: str = "video") -> str:
    """Build a stable split identity from a sample's metadata.

    Track IDs are qualified by video ID because tracker counters are commonly
    reset for each video; track_1 in two videos denotes two individuals.
    """
    metadata = getattr(sample, "metadata", None) or {}
    if group_by in {"video", "video_id"}:
        video_id = metadata.get("video_id")
        if video_id is None or not str(video_id).strip():
            raise ValueError(f"Sample {getattr(sample, 'sample_id', '<unknown>')!r} has no video_id.")
        return str(video_id)

    if group_by in {"track", "track_id"}:
        video_id = metadata.get("video_id")
        track_id = metadata.get("track_id")
        if video_id is None or not str(video_id).strip():
            raise ValueError("Track-level splitting requires metadata.video_id.")
        if track_id is None:
            raise ValueError("Track-level splitting requires metadata.track_id.")
        canonical_video_id = str(video_id).strip()
        return f"{canonical_video_id}::track::{normalize_track_id(track_id)}"

    raise ValueError("group_by must be 'video'/'video_id' or 'track'/'track_id'.")


def split_group_ids(
    groups: Sequence[str],
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    group_labels: dict[str, Sequence[str]] | None = None,
) -> SplitGroups:
    """Divide unique group IDs, optionally balancing behavior labels.

    Without group_labels this retains the legacy seeded random split.
    When labels are supplied, each group remains indivisible and the shared
    stratified assignment routine is used.
    """
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 for value in ratios):
        raise ValueError("All split ratios must be positive numbers.")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("train_ratio + validation_ratio + test_ratio must equal 1.0.")

    unique = sorted({str(group) for group in groups})
    if len(unique) < 3:
        raise ValueError("At least three distinct groups are required for splitting.")

    if group_labels is not None:
        normalized_labels = {
            str(group): labels for group, labels in group_labels.items()
        }
        if set(normalized_labels) != set(unique):
            raise ValueError(
                "group_labels must contain exactly one entry for every input group."
            )
        assignment = stratified_group_assignment(
            normalized_labels,
            ratios,
            seed=seed,
        )
        result = SplitGroups(
            train=tuple(sorted(group for group, split_index in assignment.items() if split_index == 0)),
            validation=tuple(sorted(group for group, split_index in assignment.items() if split_index == 1)),
            test=tuple(sorted(group for group, split_index in assignment.items() if split_index == 2)),
        )
        result.validate()
        return result

    import random

    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    n_groups = len(shuffled)
    n_train = max(1, int(n_groups * train_ratio))
    n_validation = max(1, int(n_groups * validation_ratio))
    if n_train + n_validation >= n_groups:
        n_train = n_groups - 2
        n_validation = 1

    result = SplitGroups(
        train=tuple(sorted(shuffled[:n_train])),
        validation=tuple(sorted(shuffled[n_train:n_train + n_validation])),
        test=tuple(sorted(shuffled[n_train + n_validation:])),
    )
    result.validate()
    return result

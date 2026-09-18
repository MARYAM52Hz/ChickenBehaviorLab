from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class FeatureDiagnosticRecord:
    """
    Feature-level diagnostic information for one prediction.

    The diagnostic layer summarizes the graph features used by
    the model without changing the original prediction.
    """

    sample_id: str
    video_id: str
    track_id: int
    start_frame: int
    end_frame: int

    true_behavior: str
    predicted_behavior: str
    confidence: float

    is_error: bool

    mean_motion_energy: float
    mean_node_feature_magnitude: float
    mean_edge_feature_magnitude: float

    valid_keypoint_ratio: float
    valid_node_ratio: float

    mean_velocity: float
    mean_acceleration: float

    mean_joint_angle: float | None
    std_joint_angle: float | None

    mean_body_orientation: float | None
    std_body_orientation: float | None

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError(
                "sample_id cannot be empty."
            )

        if not self.video_id:
            raise ValueError(
                "video_id cannot be empty."
            )

        if self.track_id < 0:
            raise ValueError(
                "track_id cannot be negative."
            )

        if self.start_frame < 0:
            raise ValueError(
                "start_frame cannot be negative."
            )

        if self.end_frame < self.start_frame:
            raise ValueError(
                "end_frame must be greater than or equal "
                "to start_frame."
            )

        if not self.true_behavior:
            raise ValueError(
                "true_behavior cannot be empty."
            )

        if not self.predicted_behavior:
            raise ValueError(
                "predicted_behavior cannot be empty."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0.0 and 1.0."
            )

        bounded_values = {
            "valid_keypoint_ratio": (
                self.valid_keypoint_ratio
            ),
            "valid_node_ratio": (
                self.valid_node_ratio
            ),
        }

        for name, value in bounded_values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0.0 and 1.0."
                )

        numeric_values = {
            "mean_motion_energy": self.mean_motion_energy,
            "mean_node_feature_magnitude": (
                self.mean_node_feature_magnitude
            ),
            "mean_edge_feature_magnitude": (
                self.mean_edge_feature_magnitude
            ),
            "mean_velocity": self.mean_velocity,
            "mean_acceleration": self.mean_acceleration,
        }

        for name, value in numeric_values.items():
            if not math.isfinite(value):
                raise ValueError(
                    f"{name} must be finite."
                )

        optional_values = {
            "mean_joint_angle": self.mean_joint_angle,
            "std_joint_angle": self.std_joint_angle,
            "mean_body_orientation": (
                self.mean_body_orientation
            ),
            "std_body_orientation": (
                self.std_body_orientation
            ),
        }

        for name, value in optional_values.items():
            if value is not None and not math.isfinite(value):
                raise ValueError(
                    f"{name} must be finite or None."
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


class FeatureDiagnosticAnalyzer:
    """
    Analyze feature characteristics of correct and incorrect
    behavior predictions.

    This class is intentionally model-agnostic.
    """

    def __init__(
        self,
        records: list[FeatureDiagnosticRecord],
    ) -> None:
        self.records = list(records)
        self._validate()

    def _validate(self) -> None:
        sample_ids: set[str] = set()

        for record in self.records:
            record.validate()

            if record.sample_id in sample_ids:
                raise ValueError(
                    f"Duplicate sample_id: "
                    f"{record.sample_id}"
                )

            sample_ids.add(record.sample_id)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def errors(self) -> list[FeatureDiagnosticRecord]:
        return [
            record
            for record in self.records
            if record.is_error
        ]

    @property
    def correct(
        self,
    ) -> list[FeatureDiagnosticRecord]:
        return [
            record
            for record in self.records
            if not record.is_error
        ]

    @staticmethod
    def _mean(
        records: list[FeatureDiagnosticRecord],
        field_name: str,
    ) -> float | None:
        if not records:
            return None

        values = [
            float(getattr(record, field_name))
            for record in records
        ]

        return float(np.mean(values))

    def compare_error_vs_correct(
        self,
    ) -> dict[str, Any]:
        feature_names = [
            "mean_motion_energy",
            "mean_node_feature_magnitude",
            "mean_edge_feature_magnitude",
            "valid_keypoint_ratio",
            "valid_node_ratio",
            "mean_velocity",
            "mean_acceleration",
            "mean_joint_angle",
            "std_joint_angle",
            "mean_body_orientation",
            "std_body_orientation",
        ]

        comparison: dict[str, Any] = {
            "num_error_samples": len(self.errors),
            "num_correct_samples": len(self.correct),
            "features": {},
        }

        for feature_name in feature_names:
            error_mean = self._mean(
                self.errors,
                feature_name,
            )

            correct_mean = self._mean(
                self.correct,
                feature_name,
            )

            difference = None

            if (
                error_mean is not None
                and correct_mean is not None
            ):
                difference = error_mean - correct_mean

            comparison["features"][feature_name] = {
                "error_mean": error_mean,
                "correct_mean": correct_mean,
                "difference": difference,
            }

        return comparison

    def compare_by_true_behavior(
        self,
    ) -> list[dict[str, Any]]:
        behaviors = sorted(
            {
                record.true_behavior
                for record in self.records
            }
        )

        results = []

        for behavior in behaviors:
            behavior_records = [
                record
                for record in self.records
                if record.true_behavior == behavior
            ]

            errors = [
                record
                for record in behavior_records
                if record.is_error
            ]

            correct = [
                record
                for record in behavior_records
                if not record.is_error
            ]

            feature_names = [
                "mean_motion_energy",
                "mean_node_feature_magnitude",
                "mean_edge_feature_magnitude",
                "valid_keypoint_ratio",
                "valid_node_ratio",
                "mean_velocity",
                "mean_acceleration",
            ]

            feature_comparison = {}

            for feature_name in feature_names:
                error_mean = self._mean(
                    errors,
                    feature_name,
                )

                correct_mean = self._mean(
                    correct,
                    feature_name,
                )

                difference = None

                if (
                    error_mean is not None
                    and correct_mean is not None
                ):
                    difference = (
                        error_mean
                        - correct_mean
                    )

                feature_comparison[
                    feature_name
                ] = {
                    "error_mean": error_mean,
                    "correct_mean": correct_mean,
                    "difference": difference,
                }

            results.append(
                {
                    "behavior": behavior,
                    "total": len(
                        behavior_records
                    ),
                    "errors": len(errors),
                    "correct": len(correct),
                    "error_rate": (
                        len(errors)
                        / len(behavior_records)
                        if behavior_records
                        else 0.0
                    ),
                    "features": feature_comparison,
                }
            )

        return results

    def suspicious_error_patterns(
        self,
        motion_threshold: float = 0.10,
        keypoint_ratio_threshold: float = 0.60,
    ) -> list[dict[str, Any]]:
        """
        Identify errors with low motion and/or poor keypoint
        availability.

        These are diagnostic flags, not scientific conclusions.
        """

        if motion_threshold < 0:
            raise ValueError(
                "motion_threshold must be non-negative."
            )

        if not 0.0 <= keypoint_ratio_threshold <= 1.0:
            raise ValueError(
                "keypoint_ratio_threshold must be "
                "between 0.0 and 1.0."
            )

        results = []

        for record in self.errors:
            reasons = []

            if (
                record.mean_motion_energy
                <= motion_threshold
            ):
                reasons.append(
                    "low_motion_energy"
                )

            if (
                record.valid_keypoint_ratio
                <= keypoint_ratio_threshold
            ):
                reasons.append(
                    "low_keypoint_visibility"
                )

            if not reasons:
                continue

            results.append(
                {
                    "sample_id": record.sample_id,
                    "true_behavior": (
                        record.true_behavior
                    ),
                    "predicted_behavior": (
                        record.predicted_behavior
                    ),
                    "confidence": record.confidence,
                    "reasons": reasons,
                }
            )

        return results

    def summary(self) -> dict[str, Any]:
        return {
            "num_records": len(self.records),
            "num_errors": len(self.errors),
            "num_correct": len(self.correct),
            "error_vs_correct": (
                self.compare_error_vs_correct()
            ),
            "by_true_behavior": (
                self.compare_by_true_behavior()
            ),
            "suspicious_error_patterns": (
                self.suspicious_error_patterns()
            ),
        }

    def save_json(
        self,
        path: str | Path,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "records": [
                record.to_dict()
                for record in self.records
            ],
            "summary": self.summary(),
        }

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
            )

    @classmethod
    def load_json(
        cls,
        path: str | Path,
    ) -> "FeatureDiagnosticAnalyzer":
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Feature diagnostic file not found: "
                f"{path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if "records" not in data:
            raise ValueError(
                "JSON must contain a 'records' field."
            )

        records = [
            FeatureDiagnosticRecord(
                sample_id=str(
                    item["sample_id"]
                ),
                video_id=str(
                    item["video_id"]
                ),
                track_id=int(
                    item["track_id"]
                ),
                start_frame=int(
                    item["start_frame"]
                ),
                end_frame=int(
                    item["end_frame"]
                ),
                true_behavior=str(
                    item["true_behavior"]
                ),
                predicted_behavior=str(
                    item["predicted_behavior"]
                ),
                confidence=float(
                    item["confidence"]
                ),
                is_error=bool(
                    item["is_error"]
                ),
                mean_motion_energy=float(
                    item[
                        "mean_motion_energy"
                    ]
                ),
                mean_node_feature_magnitude=float(
                    item[
                        "mean_node_feature_magnitude"
                    ]
                ),
                mean_edge_feature_magnitude=float(
                    item[
                        "mean_edge_feature_magnitude"
                    ]
                ),
                valid_keypoint_ratio=float(
                    item[
                        "valid_keypoint_ratio"
                    ]
                ),
                valid_node_ratio=float(
                    item[
                        "valid_node_ratio"
                    ]
                ),
                mean_velocity=float(
                    item["mean_velocity"]
                ),
                mean_acceleration=float(
                    item["mean_acceleration"]
                ),
                mean_joint_angle=(
                    float(
                        item["mean_joint_angle"]
                    )
                    if item["mean_joint_angle"]
                    is not None
                    else None
                ),
                std_joint_angle=(
                    float(
                        item["std_joint_angle"]
                    )
                    if item["std_joint_angle"]
                    is not None
                    else None
                ),
                mean_body_orientation=(
                    float(
                        item[
                            "mean_body_orientation"
                        ]
                    )
                    if item[
                        "mean_body_orientation"
                    ] is not None
                    else None
                ),
                std_body_orientation=(
                    float(
                        item[
                            "std_body_orientation"
                        ]
                    )
                    if item[
                        "std_body_orientation"
                    ] is not None
                    else None
                ),
            )
            for item in data["records"]
        ]

        return cls(records)

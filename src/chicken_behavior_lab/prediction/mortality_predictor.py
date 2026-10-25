
"""Validated wrapper around a separately trained binary probabilistic estimator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping, Sequence


class MortalityPredictionError(ValueError):
    """Input or estimator output violates the prediction contract."""


@dataclass(frozen=True)
class MortalityPrediction:
    """One model prediction; not a diagnosis or causal explanation."""

    sample_id: str
    risk_probability: float
    risk_label: str


class MortalityPredictor:
    """Wrap a fitted estimator exposing classes_ and predict_proba(X).

    This class does not train a model. Predictive validity and threshold
    selection must be established using representative, labelled,
    held-out data.
    """

    def __init__(
        self,
        estimator: Any,
        feature_names: Sequence[str],
        *,
        positive_label: Any = 1,
        threshold: float = 0.5,
        risk_label: str = "higher_modelled_risk",
        lower_label: str = "lower_modelled_risk",
    ) -> None:
        if estimator is None or not callable(
            getattr(estimator, "predict_proba", None)
        ):
            raise MortalityPredictionError(
                "estimator must expose predict_proba()."
            )

        names = tuple(feature_names)

        if not names or any(
            not isinstance(name, str) or not name.strip()
            for name in names
        ):
            raise MortalityPredictionError(
                "feature_names must contain non-empty strings."
            )

        if len(set(names)) != len(names):
            raise MortalityPredictionError(
                "feature_names must be unique."
            )

        if isinstance(threshold, bool) or not isinstance(
            threshold, Real
        ):
            raise MortalityPredictionError(
                "threshold must be a finite number in [0, 1]."
            )

        threshold = float(threshold)

        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise MortalityPredictionError(
                "threshold must be a finite number in [0, 1]."
            )

        classes = getattr(estimator, "classes_", None)

        if classes is None or len(classes) != 2:
            raise MortalityPredictionError(
                "estimator.classes_ must contain exactly two classes."
            )

        classes = list(classes)

        if positive_label not in classes:
            raise MortalityPredictionError(
                "positive_label is absent from estimator.classes_."
            )

        if (
            not isinstance(risk_label, str)
            or not risk_label.strip()
            or not isinstance(lower_label, str)
            or not lower_label.strip()
            or risk_label == lower_label
        ):
            raise MortalityPredictionError(
                "risk labels must be different, non-empty strings."
            )

        self.estimator = estimator
        self.feature_names = names
        self.positive_label = positive_label
        self.threshold = threshold
        self.risk_label = risk_label
        self.lower_label = lower_label
        self._positive_column = classes.index(positive_label)

    def _prepare_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> list[list[float]]:
        if (
            isinstance(rows, (str, bytes))
            or not isinstance(rows, Sequence)
            or not rows
        ):
            raise MortalityPredictionError(
                "rows must be a non-empty sequence of mappings."
            )

        expected = set(self.feature_names)
        matrix: list[list[float]] = []

        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise MortalityPredictionError(
                    f"row {row_index} must be a mapping."
                )

            actual = set(row)

            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)

                raise MortalityPredictionError(
                    f"row {row_index} feature mismatch; "
                    f"missing={missing}, extra={extra}."
                )

            values: list[float] = []

            for name in self.feature_names:
                value = row[name]

                if isinstance(value, bool) or not isinstance(value, Real):
                    raise MortalityPredictionError(
                        f"row {row_index} feature {name!r} "
                        "must be numeric."
                    )

                value = float(value)

                if not math.isfinite(value):
                    raise MortalityPredictionError(
                        f"row {row_index} feature {name!r} "
                        "must be finite."
                    )

                values.append(value)

            matrix.append(values)

        return matrix

    def predict(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        sample_ids: Sequence[str] | None = None,
    ) -> list[MortalityPrediction]:
        """Return positive-class probabilities and threshold-based labels."""

        matrix = self._prepare_rows(rows)

        if sample_ids is None:
            ids = [str(index) for index in range(len(matrix))]
        else:
            if (
                isinstance(sample_ids, (str, bytes))
                or len(sample_ids) != len(matrix)
            ):
                raise MortalityPredictionError(
                    "sample_ids length must match rows."
                )

            ids = list(sample_ids)

            if any(
                not isinstance(sample_id, str) or not sample_id.strip()
                for sample_id in ids
            ):
                raise MortalityPredictionError(
                    "sample_ids must be non-empty strings."
                )

        try:
            probabilities = self.estimator.predict_proba(matrix)

            if len(probabilities) != len(matrix):
                raise MortalityPredictionError(
                    "estimator returned the wrong number of rows."
                )

        except MortalityPredictionError:
            raise
        except Exception as exc:
            raise MortalityPredictionError(
                f"predict_proba failed or returned invalid output: {exc}"
            ) from exc

        results: list[MortalityPrediction] = []

        for index, probability_row in enumerate(probabilities):
            try:
                if len(probability_row) != 2:
                    raise ValueError(
                        "expected two class probabilities"
                    )

                probability = float(
                    probability_row[self._positive_column]
                )

            except (TypeError, ValueError, IndexError) as exc:
                raise MortalityPredictionError(
                    f"invalid probability row {index}."
                ) from exc

            if (
                not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise MortalityPredictionError(
                    f"probability at row {index} must be in [0, 1]."
                )

            results.append(
                MortalityPrediction(
                    sample_id=ids[index],
                    risk_probability=probability,
                    risk_label=(
                        self.risk_label
                        if probability >= self.threshold
                        else self.lower_label
                    ),
                )
            )

        return results

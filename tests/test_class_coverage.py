from __future__ import annotations

import pytest

from chicken_behavior_lab.dataset.preflight import (
    inspect_class_coverage,
    validate_class_coverage,
)


def test_class_coverage_is_valid() -> None:
    """Validation/test classes are all represented in training."""

    from chicken_behavior_lab.dataset.sample import GraphSample

    train = [
        GraphSample(
            graph=None,
            label=0,
            behavior_id="feeding",
            sample_id="train_1",
        ),
        GraphSample(
            graph=None,
            label=1,
            behavior_id="walking",
            sample_id="train_2",
        ),
    ]

    validation = [
        GraphSample(
            graph=None,
            label=0,
            behavior_id="feeding",
            sample_id="validation_1",
        ),
    ]

    test = [
        GraphSample(
            graph=None,
            label=1,
            behavior_id="walking",
            sample_id="test_1",
        ),
    ]

    report = inspect_class_coverage(
        train,
        validation,
        test,
    )

    assert report.is_valid
    assert report.validation_only_classes == set()
    assert report.test_only_classes == set()

    validated = validate_class_coverage(
        train,
        validation,
        test,
    )

    assert validated.is_valid


def test_validation_only_class_is_rejected() -> None:
    """A class appearing only in validation must be rejected."""

    from chicken_behavior_lab.dataset.sample import GraphSample

    train = [
        GraphSample(
            graph=None,
            label=0,
            behavior_id="feeding",
            sample_id="train_1",
        ),
    ]

    validation = [
        GraphSample(
            graph=None,
            label=1,
            behavior_id="walking",
            sample_id="validation_1",
        ),
    ]

    test = [
        GraphSample(
            graph=None,
            label=0,
            behavior_id="feeding",
            sample_id="test_1",
        ),
    ]

    report = inspect_class_coverage(
        train,
        validation,
        test,
    )

    assert report.validation_only_classes == {
        "walking"
    }

    with pytest.raises(
        ValueError,
        match="Validation split contains behavior classes",
    ):
        validate_class_coverage(
            train,
            validation,
            test,
        )


def test_test_only_class_is_rejected() -> None:
    """A class appearing only in test must be rejected."""

    from chicken_behavior_lab.dataset.sample import GraphSample

    train = [
        GraphSample(
            graph=None,
            label=0,
            behavior_id="feeding",
            sample_id="train_1",
        ),
    ]

    validation = [
        GraphSample(
            graph=None,
            label=0,
            behavior_id="feeding",
            sample_id="validation_1",
        ),
    ]

    test = [
        GraphSample(
            graph=None,
            label=1,
            behavior_id="aggression",
            sample_id="test_1",
        ),
    ]

    report = inspect_class_coverage(
        train,
        validation,
        test,
    )

    assert report.test_only_classes == {
        "aggression"
    }

    with pytest.raises(
        ValueError,
        match="Test split contains behavior classes",
    ):
        validate_class_coverage(
            train,
            validation,
            test,
        )

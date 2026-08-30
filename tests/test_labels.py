"""
Tests for ChickenBehaviorLab behavior labels.
"""

import pytest

from chicken_behavior_lab.annotations import (
    BehaviorLabel,
    BehaviorLabelEncoder,
)


def create_encoder():

    labels = [
        BehaviorLabel(
            behavior_id="feeding",
            name="Feeding",
        ),
        BehaviorLabel(
            behavior_id="walking",
            name="Walking",
        ),
        BehaviorLabel(
            behavior_id="standing",
            name="Standing",
        ),
    ]

    return BehaviorLabelEncoder(
        labels
    )


def test_encode():

    encoder = create_encoder()

    assert (
        encoder.encode("feeding")
        == 0
    )

    assert (
        encoder.encode("walking")
        == 1
    )

    assert (
        encoder.encode("standing")
        == 2
    )


def test_decode():

    encoder = create_encoder()

    assert (
        encoder.decode(0)
        == "feeding"
    )

    assert (
        encoder.decode(1)
        == "walking"
    )

    assert (
        encoder.decode(2)
        == "standing"
    )


def test_num_classes():

    encoder = create_encoder()

    assert (
        encoder.num_classes
        == 3
    )


def test_unknown_behavior():

    encoder = create_encoder()

    with pytest.raises(
        KeyError
    ):

        encoder.encode(
            "unknown_behavior"
        )


def test_unknown_index():

    encoder = create_encoder()

    with pytest.raises(
        KeyError
    ):

        encoder.decode(99)


def test_duplicate_behavior_ids():

    labels = [
        BehaviorLabel(
            behavior_id="feeding",
            name="Feeding",
        ),
        BehaviorLabel(
            behavior_id="feeding",
            name="Duplicate Feeding",
        ),
    ]

    with pytest.raises(
        ValueError
    ):

        BehaviorLabelEncoder(
            labels
        )

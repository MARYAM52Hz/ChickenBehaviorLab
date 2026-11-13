from __future__ import annotations

from dataclasses import dataclass

import pytest

from chicken_behavior_lab.dataset.split_diagnostics import inspect_split_quality


@dataclass
class Sample:
    sample_id: str
    behavior_id: str
    metadata: dict


def test_track_level_diagnostics_require_video_scope():
    train = [Sample("train", "feeding", {"track_id": "track_1"})]
    validation = [Sample("validation", "feeding", {"video_id": "video_2", "track_id": "track_1"})]
    test = [Sample("test", "feeding", {"video_id": "video_3", "track_id": "track_1"})]

    with pytest.raises(ValueError, match="required for track-level split diagnostics"):
        inspect_split_quality(
            train,
            validation,
            test,
            group_key="track_id",
        )


def test_track_level_diagnostics_include_video_in_group_identity():
    train = [Sample("train", "feeding", {"video_id": "video_1", "track_id": "track_1"})]
    validation = [Sample("validation", "feeding", {"video_id": "video_2", "track_id": "track_1"})]
    test = [Sample("test", "feeding", {"video_id": "video_3", "track_id": "track_1"})]

    report = inspect_split_quality(
        train,
        validation,
        test,
        group_key="track_id",
    )

    assert report.group_counts == {"train": 1, "validation": 1, "test": 1}
    assert not report.has_group_leakage
    assert report.class_group_counts == {"feeding": 3}
    assert report.class_split_coverage == {
        "feeding": {"train": True, "validation": True, "test": True}
    }

def test_diagnostics_explain_single_group_class_coverage():
    train = [
        Sample("train-feeding", "feeding", {"video_id": "video_1", "track_id": "track_1"}),
        Sample("train-rare", "dust_bathing", {"video_id": "video_1", "track_id": "track_1"}),
    ]
    validation = [
        Sample("validation-feeding", "feeding", {"video_id": "video_2", "track_id": "track_1"})
    ]
    test = [
        Sample("test-feeding", "feeding", {"video_id": "video_3", "track_id": "track_1"})
    ]

    report = inspect_split_quality(train, validation, test)

    assert report.class_group_counts == {"dust_bathing": 1, "feeding": 3}
    assert report.class_split_coverage["dust_bathing"] == {
        "train": True,
        "validation": False,
        "test": False,
    }
    assert report.class_split_coverage["feeding"] == {
        "train": True,
        "validation": True,
        "test": True,
    }
    # A class confined to one independent video is not a leakage or
    # unknown-class error when it is retained in training.
    assert report.is_valid
    assert not report.has_unknown_classes


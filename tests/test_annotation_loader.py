"""
Tests for ChickenBehaviorLab AnnotationLoader.
"""

import json

import pytest

from chicken_behavior_lab.annotations import (
    AnnotationLoader,
)


def create_annotation_file(
    tmp_path,
):
    """
    Create a temporary annotation JSON file.
    """

    data = {
        "schema_version": "0.1.0",
        "dataset_id": "test_dataset",
        "annotations": [
            {
                "annotation_id": "ann_001",
                "video_id": "video_001",
                "track_id": 1,
                "behavior_id": "feeding",
                "start_frame": 100,
                "end_frame": 120,
                "annotator": "test",
                "confidence": 0.95,
                "notes": "Test annotation.",
            },
            {
                "annotation_id": "ann_002",
                "video_id": "video_001",
                "track_id": 1,
                "behavior_id": "walking",
                "start_frame": 121,
                "end_frame": 150,
            },
        ],
    }

    path = (
        tmp_path
        / "annotations.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
        )

    return path


def test_annotation_loader(
    tmp_path,
):

    path = create_annotation_file(
        tmp_path
    )

    loader = AnnotationLoader()

    annotation_set = loader.load(
        path
    )

    assert len(
        annotation_set
    ) == 2

    first = annotation_set.annotations[0]

    assert (
        first.annotation_id
        == "ann_001"
    )

    assert (
        first.video_id
        == "video_001"
    )

    assert (
        first.track_id
        == 1
    )

    assert (
        first.behavior_id
        == "feeding"
    )

    assert (
        first.start_frame
        == 100
    )

    assert (
        first.end_frame
        == 120
    )

    assert (
        first.num_frames
        == 21
    )


def test_missing_file():

    loader = AnnotationLoader()

    with pytest.raises(
        FileNotFoundError
    ):

        loader.load(
            "does_not_exist.json"
        )


def test_invalid_extension(
    tmp_path,
):

    path = (
        tmp_path
        / "annotations.txt"
    )

    path.write_text(
        "invalid",
        encoding="utf-8",
    )

    loader = AnnotationLoader()

    with pytest.raises(
        ValueError,
        match="Only JSON",
    ):

        loader.load(
            path
        )


def test_missing_annotations_field(
    tmp_path,
):

    path = (
        tmp_path
        / "invalid.json"
    )

    path.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0"
            }
        ),
        encoding="utf-8",
    )

    loader = AnnotationLoader()

    with pytest.raises(
        ValueError,
        match="annotations",
    ):

        loader.load(
            path
        )

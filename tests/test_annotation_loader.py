"""Tests for the versioned JSON AnnotationLoader."""

import json

import pytest

from chicken_behavior_lab.annotations.loader import AnnotationLoader, AnnotationLoadError


def create_annotation_file(tmp_path, *, schema_version="1.0"):
    data = {
        "schema_version": schema_version,
        "video_id": "video_001",
        "annotations": [
            {
                "annotation_id": "ann_001",
                "track_id": 1,
                "behavior_id": "feeding",
                "start_frame": 100,
                "end_frame": 120,
                "annotator_id": "annotator_01",
                "confidence": 0.95,
                "attributes": {"note": "Test annotation."},
            },
            {
                "annotation_id": "ann_002",
                "track_id": "track_1",
                "behavior_id": "walking",
                "start_frame": 121,
                "end_frame": 150,
            },
        ],
        "metadata": {"origin": "test"},
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_annotation_loader_reads_video_scoped_schema(tmp_path):
    path = create_annotation_file(tmp_path)
    annotation_set = AnnotationLoader().load(path)

    assert annotation_set.video_id == "video_001"
    assert len(annotation_set.annotations) == 2
    first = annotation_set.annotations[0]
    assert first.annotation_id == "ann_001"
    assert first.track_id == 1
    assert first.behavior_id == "feeding"
    assert first.start_frame == 100
    assert first.end_frame == 120
    assert first.duration_frames == 21
    assert first.annotator_id == "annotator_01"
    assert first.attributes == {"note": "Test annotation."}


def test_missing_file_raises_loader_error():
    with pytest.raises(AnnotationLoadError, match="does not exist"):
        AnnotationLoader().load("does_not_exist.json")


def test_invalid_extension_is_rejected_before_json_parsing(tmp_path):
    path = tmp_path / "annotations.txt"
    path.write_text("invalid", encoding="utf-8")
    with pytest.raises(AnnotationLoadError, match="Only JSON"):
        AnnotationLoader().load(path)


def test_missing_annotations_field_is_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"schema_version": "1.0", "video_id": "video_001"}), encoding="utf-8")
    with pytest.raises(AnnotationLoadError, match="annotations"):
        AnnotationLoader().load(path)


def test_unsupported_schema_version_is_rejected(tmp_path):
    path = create_annotation_file(tmp_path, schema_version="0.1.0")
    with pytest.raises(AnnotationLoadError, match="Unsupported schema_version"):
        AnnotationLoader().load(path)


def test_loader_save_round_trip(tmp_path):
    source = create_annotation_file(tmp_path)
    loader = AnnotationLoader()
    loaded = loader.load(source)
    output = tmp_path / "saved" / "annotations.json"
    loader.save(loaded, output)
    restored = loader.load(output)
    assert restored.to_dict() == loaded.to_dict()


def test_empty_annotations_can_be_disallowed(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(
        json.dumps({"schema_version": "1.0", "video_id": "video_001", "annotations": []}),
        encoding="utf-8",
    )
    with pytest.raises(AnnotationLoadError, match="no annotations"):
        AnnotationLoader(allow_empty_annotations=False).load(path)

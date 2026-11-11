"""Tests for media asset inventory and provenance contracts."""
import hashlib
import json

import pytest

from chicken_behavior_lab.io.media_manifest_io import MediaManifestIO
from chicken_behavior_lab.media.manifest import (
    MediaAsset, MediaManifest, MediaProperties, sha256_file,
)


def test_register_file_records_relative_path_size_and_checksum(tmp_path):
    root = tmp_path / "workspace"
    source = root / "raw" / "videos" / "clip.strange"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source media bytes; no decoder assumption")

    asset = MediaAsset.from_file(
        source, asset_id="video_001", media_type="video", media_root=root
    )
    assert asset.relative_path == "raw/videos/clip.strange"
    assert asset.original_filename == "clip.strange"
    assert asset.file_size_bytes == source.stat().st_size
    assert asset.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert str(root) not in json.dumps(asset.to_dict())


def test_sha256_file_streams_and_rejects_invalid_paths(tmp_path):
    source = tmp_path / "sample.bin"
    source.write_bytes(b"0123456789")
    assert sha256_file(source, chunk_size=3) == hashlib.sha256(b"0123456789").hexdigest()
    with pytest.raises(ValueError, match="positive integer"):
        sha256_file(source, chunk_size=0)
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "missing.bin")


def test_register_file_rejects_file_outside_media_root(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError, match="inside media_root"):
        MediaAsset.from_file(outside, asset_id="video_001", media_type="video", media_root=root)


@pytest.mark.parametrize(
    "relative_path",
    ["../escape.mp4", "/absolute/video.mp4", "raw\\\\video.mp4", "C:/data/video.mp4"],
)
def test_media_asset_rejects_nonportable_paths(relative_path):
    with pytest.raises(ValueError):
        MediaAsset(
            asset_id="video_001", media_type="video",
            original_filename="video.mp4", relative_path=relative_path,
        )


def test_media_asset_can_reference_external_storage_without_local_path():
    asset = MediaAsset(
        asset_id="video_002", media_type="video", original_filename="source-video",
        source_reference={"provider": "external_storage", "reference_id": "opaque-key"},
    )
    assert asset.relative_path is None
    assert asset.source_reference["provider"] == "external_storage"


def test_properties_preserve_rational_frame_rate_and_validate_pairs():
    props = MediaProperties(
        width=1920, height=1080, frame_count=300,
        average_frame_rate_num=30000, average_frame_rate_den=1001,
        time_base_num=1, time_base_den=90000, duration_seconds=10.01,
        format_name="matroska",
    )
    assert props.average_frame_rate == pytest.approx(29.97002997)
    with pytest.raises(ValueError, match="set together"):
        MediaProperties(average_frame_rate_num=30)


def test_image_assets_reject_video_only_properties():
    with pytest.raises(ValueError, match="video-only"):
        MediaAsset(
            asset_id="image_001", media_type="image", original_filename="still.jpg",
            relative_path="raw/images/still.jpg", properties=MediaProperties(frame_count=1),
        )


def test_manifest_rejects_duplicate_ids_and_case_insensitive_paths():
    first = MediaAsset("video_001", "video", "a.mp4", relative_path="raw/a.mp4")
    duplicate_id = MediaAsset("video_001", "video", "b.mp4", relative_path="raw/b.mp4")
    with pytest.raises(ValueError, match="asset_id"):
        MediaManifest((first, duplicate_id))
    duplicate_path = MediaAsset("video_002", "video", "a.mp4", relative_path="RAW/a.mp4")
    with pytest.raises(ValueError, match="relative_path"):
        MediaManifest((first, duplicate_path))


def test_manifest_checks_derivative_parent_and_cycles():
    orphan = MediaAsset(
        "derived_001", "video", "converted.mp4", relative_path="processed/converted.mp4",
        role="derivative", parent_asset_id="video_missing",
    )
    with pytest.raises(ValueError, match="missing"):
        MediaManifest((orphan,))
    child = MediaAsset(
        "derived_001", "video", "converted.mp4", relative_path="processed/converted.mp4",
        role="derivative", parent_asset_id="video_001",
    )
    parent = MediaAsset(
        "video_001", "video", "original.mp4", relative_path="raw/original.mp4",
        role="derivative", parent_asset_id="derived_001",
    )
    with pytest.raises(ValueError, match="cycle"):
        MediaManifest((child, parent))


def test_media_manifest_round_trip_and_asset_lookup(tmp_path):
    original = MediaAsset(
        "video_001", "video", "original.mov", relative_path="raw/videos/original.mov",
        sha256="A" * 64, file_size_bytes=100,
        properties=MediaProperties(width=1280, height=720),
    )
    derivative = MediaAsset(
        "video_001_mp4", "video", "working.mp4", relative_path="processed/working.mp4",
        role="derivative", parent_asset_id="video_001",
        properties=MediaProperties(width=1280, height=720),
        metadata={"conversion": "pending"},
    )
    image = MediaAsset(
        "image_001", "image", "still.jpg", relative_path="raw/images/still.jpg",
        properties=MediaProperties(width=2048, height=1536, format_name="jpeg"),
    )
    manifest = MediaManifest(
        (original, derivative, image), metadata={"project": "ChickenBehaviorLab"}
    )
    path = MediaManifestIO.save(manifest, tmp_path / "media_manifest.json")
    loaded = MediaManifestIO.load(path)
    assert loaded.get("video_001").sha256 == "a" * 64
    assert loaded.get("video_001_mp4").parent_asset_id == "video_001"
    assert loaded.get("image_001").media_type == "image"
    assert loaded.to_dict() == manifest.to_dict()


def test_media_manifest_io_rejects_bad_json_and_unknown_schema(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        MediaManifestIO.load(bad)
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"schema_version": "9.9", "assets": [], "metadata": {}}))
    with pytest.raises(ValueError, match="Unsupported media manifest schema"):
        MediaManifestIO.load(unknown)


def test_media_manifest_io_requires_json_extension(tmp_path):
    with pytest.raises(ValueError, match=".json"):
        MediaManifestIO.save(MediaManifest(()), tmp_path / "manifest.txt")

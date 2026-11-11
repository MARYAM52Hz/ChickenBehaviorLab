"""Versioned inventory of original and derived media assets.

This module records media identity and provenance. It deliberately does not
decode files or infer codecs and frame rates from filename extensions.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

_MEDIA_TYPES = frozenset({"video", "image"})
_ROLES = frozenset({"original", "derivative"})
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _relative_path(value: str) -> str:
    raw = _string(value, "relative_path")
    if "\\" in raw:
        raise ValueError("relative_path must use forward slashes.")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ValueError("relative_path must be relative and cannot contain '.' or '..'.")
    if path.parts and re.match(r"^[A-Za-z]:", path.parts[0]):
        raise ValueError("relative_path cannot contain a Windows drive prefix.")
    return path.as_posix()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA-256 checksum without loading the whole file into memory."""
    source = Path(path).expanduser()
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if not source.is_file():
        raise FileNotFoundError(f"Media file does not exist or is not a file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MediaProperties:
    """Known technical properties; unknown values remain None.

    Frame rate and time base use rational integer pairs, preserving exact source
    metadata where available (including non-integer rates such as 30000/1001).
    """

    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    average_frame_rate_num: int | None = None
    average_frame_rate_den: int | None = None
    time_base_num: int | None = None
    time_base_den: int | None = None
    duration_seconds: float | None = None
    format_name: str | None = None
    video_codec: str | None = None
    audio_codec: str | None = None

    def __post_init__(self) -> None:
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height must both be set or both be None.")
        if self.width is not None:
            _positive_int(self.width, "width")
            _positive_int(self.height, "height")
        for name in ("frame_count", "average_frame_rate_num", "average_frame_rate_den",
                     "time_base_num", "time_base_den"):
            value = getattr(self, name)
            if value is not None:
                _positive_int(value, name)
        if (self.average_frame_rate_num is None) != (self.average_frame_rate_den is None):
            raise ValueError("average frame-rate numerator and denominator must be set together.")
        if (self.time_base_num is None) != (self.time_base_den is None):
            raise ValueError("time-base numerator and denominator must be set together.")
        if self.duration_seconds is not None:
            if (isinstance(self.duration_seconds, bool)
                    or not isinstance(self.duration_seconds, (int, float))
                    or not isfinite(float(self.duration_seconds))
                    or self.duration_seconds < 0):
                raise ValueError("duration_seconds must be finite and non-negative.")
            object.__setattr__(self, "duration_seconds", float(self.duration_seconds))
        for name in ("format_name", "video_codec", "audio_codec"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _string(value, name))

    @property
    def average_frame_rate(self) -> float | None:
        if self.average_frame_rate_num is None:
            return None
        return self.average_frame_rate_num / self.average_frame_rate_den

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MediaProperties":
        if not isinstance(payload, Mapping):
            raise ValueError("properties must be a JSON object.")
        allowed = set(cls.__dataclass_fields__)
        extra = set(payload) - allowed
        if extra:
            raise ValueError(f"Unexpected media property field(s): {', '.join(sorted(extra))}.")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class MediaAsset:
    """One original image/video file or a derived media file."""

    asset_id: str
    media_type: Literal["video", "image"]
    original_filename: str
    relative_path: str | None = None
    source_reference: Mapping[str, str] = field(default_factory=dict)
    sha256: str | None = None
    file_size_bytes: int | None = None
    role: Literal["original", "derivative"] = "original"
    parent_asset_id: str | None = None
    properties: MediaProperties = field(default_factory=MediaProperties)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        asset_id = _string(self.asset_id, "asset_id")
        if not _ID_RE.fullmatch(asset_id) or asset_id in {".", ".."}:
            raise ValueError("asset_id may contain only letters, digits, '.', '_' and '-'.")
        object.__setattr__(self, "asset_id", asset_id)
        if self.media_type not in _MEDIA_TYPES:
            raise ValueError("media_type must be 'video' or 'image'.")
        if self.role not in _ROLES:
            raise ValueError("role must be 'original' or 'derivative'.")

        filename = _string(self.original_filename, "original_filename")
        if "/" in filename or "\\" in filename or filename in {".", ".."}:
            raise ValueError("original_filename must be a filename, not a path.")
        object.__setattr__(self, "original_filename", filename)

        if self.relative_path is not None:
            object.__setattr__(self, "relative_path", _relative_path(self.relative_path))
        source = dict(self.source_reference)
        if source and "provider" not in source:
            raise ValueError("source_reference must include a non-empty 'provider'.")
        for key, value in source.items():
            _string(key, "source_reference key")
            _string(value, f"source_reference[{key!r}]")
        object.__setattr__(self, "source_reference", source)
        if self.relative_path is None and not source:
            raise ValueError("Each asset needs a relative_path or source_reference.")

        if self.sha256 is not None:
            checksum = _string(self.sha256, "sha256").lower()
            if not _SHA256_RE.fullmatch(checksum):
                raise ValueError("sha256 must contain exactly 64 hexadecimal characters.")
            object.__setattr__(self, "sha256", checksum)
        if self.file_size_bytes is not None:
            if (isinstance(self.file_size_bytes, bool)
                    or not isinstance(self.file_size_bytes, int)
                    or self.file_size_bytes < 0):
                raise ValueError("file_size_bytes must be a non-negative integer.")

        if self.role == "original" and self.parent_asset_id is not None:
            raise ValueError("An original asset cannot have parent_asset_id.")
        if self.role == "derivative":
            parent_id = _string(self.parent_asset_id, "parent_asset_id")
            if parent_id == asset_id:
                raise ValueError("An asset cannot be its own parent.")
            object.__setattr__(self, "parent_asset_id", parent_id)

        if not isinstance(self.properties, MediaProperties):
            if isinstance(self.properties, Mapping):
                object.__setattr__(self, "properties", MediaProperties.from_dict(self.properties))
            else:
                raise TypeError("properties must be MediaProperties or a mapping.")
        if self.media_type == "image":
            video_fields = (
                self.properties.frame_count, self.properties.average_frame_rate_num,
                self.properties.average_frame_rate_den, self.properties.time_base_num,
                self.properties.time_base_den, self.properties.duration_seconds,
                self.properties.video_codec, self.properties.audio_codec,
            )
            if any(value is not None for value in video_fields):
                raise ValueError("Image assets cannot contain video-only properties.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        asset_id: str,
        media_type: Literal["video", "image"],
        media_root: str | Path,
        **kwargs: Any,
    ) -> "MediaAsset":
        """Register a file under media_root; persist only a relative path."""
        source = Path(path).expanduser().resolve(strict=True)
        root = Path(media_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"media_root is not a directory: {root}")
        if not source.is_file():
            raise ValueError(f"Media path is not a file: {source}")
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("Media file must be located inside media_root.") from exc
        return cls(
            asset_id=asset_id,
            media_type=media_type,
            original_filename=source.name,
            relative_path=relative,
            sha256=sha256_file(source),
            file_size_bytes=source.stat().st_size,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "media_type": self.media_type,
            "original_filename": self.original_filename,
            "relative_path": self.relative_path,
            "source_reference": dict(self.source_reference),
            "sha256": self.sha256,
            "file_size_bytes": self.file_size_bytes,
            "role": self.role,
            "parent_asset_id": self.parent_asset_id,
            "properties": self.properties.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MediaAsset":
        if not isinstance(payload, Mapping):
            raise ValueError("Each media asset must be a JSON object.")
        allowed = {
            "asset_id", "media_type", "original_filename", "relative_path",
            "source_reference", "sha256", "file_size_bytes", "role",
            "parent_asset_id", "properties", "metadata",
        }
        extra = set(payload) - allowed
        if extra:
            raise ValueError(f"Unexpected media asset field(s): {', '.join(sorted(extra))}.")
        values = dict(payload)
        values["properties"] = MediaProperties.from_dict(values.get("properties", {}))
        return cls(**values)


@dataclass(frozen=True, slots=True)
class MediaManifest:
    """A versioned inventory of project media assets."""

    assets: tuple[MediaAsset, ...]
    schema_version: str = "1.0"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _string(self.schema_version, "schema_version"))
        assets = tuple(self.assets)
        if any(not isinstance(item, MediaAsset) for item in assets):
            raise TypeError("assets must contain only MediaAsset instances.")
        object.__setattr__(self, "assets", assets)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("manifest metadata must be a mapping.")
        object.__setattr__(self, "metadata", dict(self.metadata))
        self.validate()

    def validate(self) -> None:
        ids = [item.asset_id for item in self.assets]
        if len(ids) != len(set(ids)):
            raise ValueError("asset_id values must be unique.")
        paths = [item.relative_path.casefold() for item in self.assets if item.relative_path]
        if len(paths) != len(set(paths)):
            raise ValueError("relative_path values must be unique.")
        by_id = {item.asset_id: item for item in self.assets}
        for item in self.assets:
            if item.parent_asset_id is not None and item.parent_asset_id not in by_id:
                raise ValueError(f"Parent asset {item.parent_asset_id!r} for {item.asset_id!r} is missing.")
        for item in self.assets:
            visited: set[str] = set()
            current = item
            while current.parent_asset_id is not None:
                if current.asset_id in visited:
                    raise ValueError("Media asset parent relationships contain a cycle.")
                visited.add(current.asset_id)
                current = by_id[current.parent_asset_id]

    def get(self, asset_id: str) -> MediaAsset:
        for item in self.assets:
            if item.asset_id == asset_id:
                return item
        raise KeyError(asset_id)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "assets": [item.to_dict() for item in self.assets],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MediaManifest":
        if not isinstance(payload, Mapping):
            raise ValueError("Media manifest JSON must contain an object.")
        extra = set(payload) - {"schema_version", "assets", "metadata"}
        if extra:
            raise ValueError(f"Unexpected manifest field(s): {', '.join(sorted(extra))}.")
        raw_assets = payload.get("assets")
        if not isinstance(raw_assets, list):
            raise ValueError("'assets' must be a JSON array.")
        return cls(
            assets=tuple(MediaAsset.from_dict(item) for item in raw_assets),
            schema_version=payload.get("schema_version", "1.0"),
            metadata=payload.get("metadata", {}),
        )

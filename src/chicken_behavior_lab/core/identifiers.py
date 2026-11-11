"""Canonical identifier handling shared across model, pipeline, and dataset layers."""

from __future__ import annotations

import re
from numbers import Integral


_TRACK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def normalize_track_id(value: str | int) -> str:
    """Return a stable string track ID.

    Integer IDs from existing annotation files are migrated to the model-layer
    convention (1 becomes track_1). Already named IDs such as track_1 and
    opaque IDs such as UUIDs are preserved. Numeric strings are treated like
    integer IDs. Booleans, empty strings, path separators, and unsafe
    path-like identifiers are rejected.
    """
    if isinstance(value, bool):
        raise ValueError("track_id must not be a boolean.")

    if isinstance(value, Integral):
        if value < 0:
            raise ValueError("track_id must be non-negative.")
        return f"track_{int(value)}"

    if not isinstance(value, str):
        raise TypeError("track_id must be a string or a non-negative integer.")

    normalized = value.strip()
    if not normalized:
        raise ValueError("track_id must not be empty.")
    if (
        not _TRACK_ID_PATTERN.fullmatch(normalized)
        or normalized in {".", ".."}
        or normalized.endswith(".")
        or normalized.split(".")[0].upper() in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError("track_id contains unsupported characters or path components.")

    # Decimal strings represent numeric tracker IDs; normalize their
    # textual form so "001" and 1 resolve to the same identity.
    if re.fullmatch(r"[0-9]+", normalized):
        return f"track_{int(normalized)}"
    return normalized

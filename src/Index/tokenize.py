"""Cell tokenization shared by ingest and query; both sides must use `tokenize_cell`."""

from __future__ import annotations

from typing import Any

_MAX_LEN = 200
_NULL_TOKENS = frozenset({"nan", "none"})
_STRIP_CHARS = ("\\", "'", '"', "\t", "\n", "\r")


def tokenize_cell(value: Any) -> str:
    """Return the canonical stored form of one cell value; step order must match the built index."""
    s = str(value).lower()
    for ch in _STRIP_CHARS:
        s = s.replace(ch, "")
    s = s.strip()[:_MAX_LEN]
    if s in _NULL_TOKENS:
        return ""
    return s

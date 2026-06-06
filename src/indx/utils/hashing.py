"""Stable content hashing for provenance, dedup, and deterministic IDs."""

from __future__ import annotations

import hashlib


def stable_hash(data: str | bytes) -> str:
    """Return a short, stable hex digest. Deterministic across runs/machines (NFR-DET-1)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]

"""`.indx` container constants (technology-selections §11: Zip + manifest)."""

from __future__ import annotations

import hashlib

MANIFEST = "manifest.json"
DOCUMENTS = "documents.jsonl"
CHUNKS = "chunks.jsonl"
RELATIONS = "relations.jsonl"

# Per-member integrity record (indx-archive.md "Integrity": per-member SHA-256 checksums).
# Stored as a dedicated member rather than a Manifest field so the integrity record can name
# the manifest itself without a self-referential digest.
CHECKSUMS = "checksums.json"
CHECKSUM_ALGO = "sha256"

# Bump deliberately on a breaking change to the on-disk layout; readers check it.
SCHEMA_VERSION = "1"


def checksum(data: str) -> str:
    """SHA-256 hex digest of a member's UTF-8 bytes (indx-archive.md, §11 packaging)."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

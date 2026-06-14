"""`.indx` container constants (technology-selections §11: Zip + manifest)."""

from __future__ import annotations

import hashlib
from pathlib import Path

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
# v2 (Feature 2, indx-of-indx) adds the additive/optional ``Manifest.children`` field; the
# member layout is otherwise unchanged, so v1 archives remain loadable.
SCHEMA_VERSION = "2"

# Every schema version the reader still accepts. Older versions stay loadable because the new
# fields are additive + optional (a v1 manifest simply has ``children == []``).
SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1", "2"})


def checksum(data: str) -> str:
    """SHA-256 hex digest of a member's UTF-8 bytes (indx-archive.md, §11 packaging)."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file's raw bytes (Feature 2 child-archive integrity pin).

    Distinct from :func:`checksum` (which digests a member's UTF-8 *text*): a ``ChildRef.sha256``
    pins the child ``.indx`` file's raw bytes, so it is computed over the file on disk.
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

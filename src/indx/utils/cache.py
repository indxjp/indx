"""Content-addressed stage cache for ``--resume`` (technical-spec §10.3).

A tiny, dependency-free JSON cache rooted under ``<out>/.indx-cache/``. Entries are keyed on
``(stage, sha256(input), component-id)`` so a re-run reuses unchanged stage outputs and skips
recomputation for unmodified files and unchanged components. The store is deliberately simple
and deterministic — one JSON file per key — which keeps it portable, diffable, and easy to
reason about (coding-standards §1.4, §14).

Scope is intentionally limited to the **Parse** and **Embed** stages, whose outputs are
serializable (a :class:`~indx.core.parsed.ParsedDoc` and a vector, respectively).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_DIRNAME = ".indx-cache"


def sha256_bytes(data: bytes) -> str:
    """Return the full hex SHA-256 of ``data`` (the content address of a cache input)."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Content-address a file by streaming its bytes (never loads the whole file at once)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Content-address a text payload (e.g. a chunk body)."""
    return sha256_bytes(text.encode("utf-8"))


class StageCache:
    """A content-addressed cache of stage outputs under ``<root>/.indx-cache/``.

    The cache is a no-op shell when ``enabled`` is ``False`` (the default, non-``--resume``
    path): ``get`` always misses and ``put`` does nothing, so the hot path pays nothing.

    Args:
        root: The output directory; the cache lives in ``root/.indx-cache``.
        enabled: Whether reads/writes actually touch disk (``--resume`` toggles this).
    """

    def __init__(self, root: Path | None, *, enabled: bool = False) -> None:
        self.enabled = enabled and root is not None
        self._dir = Path(root) / CACHE_DIRNAME if root is not None else None

    def _path(self, stage: str, input_hash: str, component_id: str) -> Path:
        assert self._dir is not None  # guarded by ``enabled``
        # Component id is folded into the address so swapping a component invalidates its
        # entries (changing the embedder invalidates only Embed, per technical-spec §10.3).
        key = sha256_text(f"{stage}\x00{input_hash}\x00{component_id}")
        return self._dir / stage / f"{key}.json"

    def get(self, stage: str, input_hash: str, component_id: str) -> Any | None:
        """Return the cached payload for the key, or ``None`` on a miss / disabled cache."""
        if not self.enabled:
            return None
        path = self._path(stage, input_hash, component_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt entry is a miss, never a crash — resume must stay best-effort.
            return None

    def put(self, stage: str, input_hash: str, component_id: str, payload: Any) -> None:
        """Persist ``payload`` (JSON-serializable) under the content-addressed key."""
        if not self.enabled:
            return
        path = self._path(stage, input_hash, component_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # sort_keys keeps the on-disk form byte-stable for identical inputs (NFR-DET-1).
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

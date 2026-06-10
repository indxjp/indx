"""Regression: JsonlWriter must write manifest.json with explicit UTF-8 encoding.

Mirrors the uniform UTF-8 contract of the rest of the output layer (the sibling
.jsonl files and IndxWriter's own manifest.json all pass ``encoding="utf-8"``).
Without it, ``Path.write_text`` falls back to the ambient locale encoding
(cp1252 / ANSI_X3.4 under ``LANG=C``), leaving one file in the output directory
written with a different encoding than its siblings.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from indx.core.knowledge_space import KnowledgeSpace, Manifest
from indx.output.jsonl_writer import JsonlWriter


def test_manifest_json_written_as_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    original = Path.write_text

    def _spy(self: Path, data: str, *args: object, **kwargs: object) -> int:
        if self.name == "manifest.json":
            captured["encoding"] = kwargs.get("encoding")
        return original(self, data, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", _spy)

    space = KnowledgeSpace(
        manifest=Manifest(source_root="/data/café", embedding_model="modèle-é"),
    )
    JsonlWriter().write(space, tmp_path)

    assert captured["encoding"] == "utf-8"

    manifest_path = tmp_path / "manifest.json"
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["source_root"] == "/data/café"
    assert loaded["embedding_model"] == "modèle-é"

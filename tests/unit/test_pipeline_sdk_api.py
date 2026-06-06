"""The documented DirectoryPipeline SDK surface runs on the offline core stack (sdk.md).

These exercise the exact snippets from the SDK reference — constructor slots, fluent
component binding, stage management, and ``run(src, out)`` — using only the zero-dependency
core components (plaintext / none / hash / jsonl) so no extra is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indx import DirectoryPipeline, KnowledgeSpace, SpaceContext
from indx.errors import RegistryError

# The offline core stack: every default that would need an extra is overridden.
OFFLINE: dict[str, object] = {
    "parser": "plaintext",
    "llm": "none",
    "embedder": "hash",
    "store": "jsonl",
    "output": "jsonl",
}


def _tree(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.md").write_text("# A\n\nFirst document body.\n")
    (root / "b.md").write_text("# B\n\nSecond document body.\n")
    return root


def _offline_toml(tmp_path: Path) -> Path:
    """An indx.toml that selects the zero-dependency core stack for every slot."""
    cfg = tmp_path / "indx.toml"
    cfg.write_text(
        "[parser]\nengine = 'plaintext'\n\n"
        "[enrich]\nllm = 'none'\nvlm = 'none'\n\n"
        "[embed]\nmodel = 'hash'\n\n"
        "[store]\nbackend = 'jsonl'\n\n"
        "[output]\nformat = 'jsonl'\n"
    )
    return cfg


def test_constructor_accepts_instance_name_or_none(tmp_path: Path) -> None:
    # sdk.md: each slot is instance | name | None; use() swaps by keyword and chains.
    # Unset slots (parser/vlm) fall back to indx.toml -> offline core here.
    cfg = _offline_toml(tmp_path)
    pipe = DirectoryPipeline(embedder="hash", config=str(cfg)).use(store="jsonl", llm="none")
    assert isinstance(pipe, DirectoryPipeline)
    names = [s.name for s in pipe.stages()]
    assert names[0] == "walk"
    assert "embed-pack" in names  # stage 6 is named embed-pack, not pack


def test_drop_enrich_and_embed_pack() -> None:
    pipe = DirectoryPipeline(**OFFLINE)  # type: ignore[arg-type]
    pipe.drop("enrich").drop("embed-pack")
    names = [s.name for s in pipe.stages()]
    assert "enrich" not in names
    assert "embed-pack" not in names


def test_drop_unknown_stage_raises() -> None:
    pipe = DirectoryPipeline(**OFFLINE)  # type: ignore[arg-type]
    with pytest.raises(RegistryError):
        pipe.drop("does-not-exist")


def test_insert_custom_stage_runs(tmp_path: Path) -> None:
    seen: list[str] = []

    class MarkerStage:
        name = "marker"

        def run(self, ctx: SpaceContext) -> SpaceContext:
            seen.append("ran")
            return ctx  # MUST return the same context

    pipe = DirectoryPipeline(**OFFLINE)  # type: ignore[arg-type]
    pipe.insert(3, MarkerStage())
    assert pipe.stages()[3].name == "marker"
    pipe.run(_tree(tmp_path / "src"))
    assert seen == ["ran"]


def test_run_with_out_writes_layout(tmp_path: Path) -> None:
    # Documented: run(src, out) seals the output layout and returns the space.
    src = _tree(tmp_path / "src")
    out = tmp_path / "ai-ready"
    pipe = DirectoryPipeline(**OFFLINE).drop("enrich")  # type: ignore[arg-type]
    space = pipe.run(src, out)
    assert isinstance(space, KnowledgeSpace)
    # jsonl writer emits its documents/chunks/manifest layout under out/.
    assert (out / "documents.jsonl").is_file()
    assert (out / "manifest.json").is_file()


def test_run_without_out_stays_in_memory(tmp_path: Path) -> None:
    src = _tree(tmp_path / "src")
    out = tmp_path / "should-not-exist"
    pipe = DirectoryPipeline(**OFFLINE)  # type: ignore[arg-type]
    space = pipe.run(src)  # out omitted -> in memory, nothing written
    assert isinstance(space, KnowledgeSpace)
    assert not out.exists()
    assert space.documents()


def test_replace_and_append_chain(tmp_path: Path) -> None:
    class NoopStage:
        name = "relate"  # replace the built-in relate stage

        def run(self, ctx: SpaceContext) -> SpaceContext:
            return ctx

    class TailStage:
        name = "tail"

        def run(self, ctx: SpaceContext) -> SpaceContext:
            return ctx

    pipe = DirectoryPipeline(**OFFLINE)  # type: ignore[arg-type]
    result = pipe.replace("relate", NoopStage()).append(TailStage())
    assert result is pipe
    names = [s.name for s in pipe.stages()]
    assert names[-1] == "tail"
    assert any(isinstance(s, NoopStage) for s in pipe.stages())

"""Pipeline cross-cutting features: strict, dry-run/plan, jobs, resume.

These exercise the orchestration-layer behavior added to :class:`DirectoryPipeline` without
touching the individual stage files. Everything runs on the zero-dependency offline stack
(plaintext / hash / jsonl) so the suite stays network-free.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indx.config import Config
from indx.core.context import SpaceContext, StageErrorRecord
from indx.core.parsed import Block
from indx.errors import StageError
from indx.pipeline import BuildPlan, DirectoryPipeline
from indx.pipeline.stages.enrich import EnrichStage
from indx.utils.cache import CACHE_DIRNAME, StageCache, sha256_text

OFFLINE = {"parser": "plaintext", "llm": "none", "embedder": "hash", "store": "jsonl"}


def _tree(root: Path) -> Path:
    (root / "a").mkdir(parents=True)
    (root / "a" / "one.md").write_text("# One\n\nAlpha beta gamma delta.\n")
    (root / "a" / "two.md").write_text("# Two\n\nEpsilon zeta eta theta.\n")
    (root / "root.txt").write_text("Top level note about onboarding.\n")
    return root


def _pipeline(out: Path | None = None, **kw: object) -> DirectoryPipeline:
    return DirectoryPipeline(**OFFLINE, out=out, **kw)  # type: ignore[arg-type]


# ── dry-run / plan ──────────────────────────────────────────────────────────


def test_plan_walks_only_and_lists_files(tmp_path: Path) -> None:
    src = _tree(tmp_path / "src")
    plan = _pipeline().plan(src)
    assert isinstance(plan, BuildPlan)
    paths = [d.path for d in plan.documents]
    assert paths == sorted(paths)  # deterministic
    assert "a/one.md" in paths and "root.txt" in paths
    assert plan.folders == sorted(plan.folders)
    assert plan.components["parser"] == "plaintext"


def test_dry_run_is_an_alias_for_plan(tmp_path: Path) -> None:
    src = _tree(tmp_path / "src")
    pipe = _pipeline()
    assert [d.path for d in pipe.dry_run(src).documents] == [
        d.path for d in pipe.plan(src).documents
    ]


def test_plan_does_not_parse_or_embed(tmp_path: Path) -> None:
    src = _tree(tmp_path / "src")
    out = tmp_path / "out"
    pipe = _pipeline(out=out)
    plan = pipe.plan(src)
    # No chunks/relations are produced, store is untouched, nothing written.
    assert plan.documents
    assert not out.exists()


# ── jobs / determinism ──────────────────────────────────────────────────────


def test_output_is_identical_regardless_of_worker_count(tmp_path: Path) -> None:
    src = _tree(tmp_path / "src")
    one = _pipeline(jobs=1).run(src)
    many = _pipeline(jobs=8).run(src)
    assert [d.id for d in one.documents()] == [d.id for d in many.documents()]
    assert [c.id for c in one.chunks] == [c.id for c in many.chunks]
    assert [c.text for c in one.chunks] == [c.text for c in many.chunks]
    assert [c.embedding for c in one.chunks] == [c.embedding for c in many.chunks]


def test_jobs_defaults_to_cpu_count(tmp_path: Path) -> None:
    import os

    pipe = _pipeline()
    assert pipe.jobs == (os.cpu_count() or 1)


def test_jobs_non_positive_falls_back(tmp_path: Path) -> None:
    assert _pipeline(jobs=0).jobs >= 1


# ── strict ──────────────────────────────────────────────────────────────────


def test_strict_promotes_skip_to_fatal(tmp_path: Path) -> None:
    """A skip recorded by a stage is promoted to a fatal StageError under strict mode."""

    class SkippingStage:
        name = "flaky"

        def run(self, ctx: SpaceContext) -> SpaceContext:
            ctx.errors.append(
                StageErrorRecord(stage="flaky", kind="skip", item="bad.md", message="boom")
            )
            return ctx

    pipe = _pipeline(strict=True)
    pipe.stages().insert(1, SkippingStage())
    with pytest.raises(StageError) as exc:
        pipe.run(_tree(tmp_path / "src"))
    assert "bad.md" in str(exc.value)
    assert "strict" in str(exc.value).lower()


def test_non_strict_keeps_skip_non_fatal(tmp_path: Path) -> None:
    class SkippingStage:
        name = "flaky"

        def run(self, ctx: SpaceContext) -> SpaceContext:
            ctx.errors.append(StageErrorRecord(stage="flaky", kind="skip", message="boom"))
            return ctx

    pipe = _pipeline(strict=False)
    pipe.stages().insert(1, SkippingStage())
    space = pipe.run(_tree(tmp_path / "src"))  # must not raise
    assert space.documents()


def test_unreadable_file_under_strict_is_fatal(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "ok.md").write_text("fine\n")

    class BoomParser:
        name = "boom"

        def parse(self, path: Path) -> object:
            raise RuntimeError("cannot parse")

    pipe = DirectoryPipeline(**OFFLINE, strict=True)  # type: ignore[arg-type]
    # Swap the parse stage's parser for one that always fails, exercising the skip→fatal path.
    pipe.stages()[1]._parser = BoomParser()  # type: ignore[attr-defined]
    with pytest.raises(StageError):
        pipe.run(src)


@pytest.mark.parametrize("jobs", [1, 4])
def test_hash_failure_between_walk_and_parse_is_a_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: int
) -> None:
    """A file that vanishes after Walk (so ``sha256_file`` raises) is skipped, not fatal.

    ``sha256_file`` opens the file and runs *outside* the parser, so when a file is deleted /
    chmod'd between the stat-only Walk and the parse fan-out, its failure must still become a
    per-file ``kind="skip"`` rather than tearing down the run (or, with jobs>1, the pool).
    """
    import indx.pipeline.pipeline as pipeline_mod

    src = tmp_path / "src"
    src.mkdir()
    (src / "good.md").write_text("# Good\n\nReadable body text.\n")
    (src / "bad.md").write_text("# Bad\n\nWill be unreadable.\n")

    real_sha256_file = pipeline_mod.sha256_file

    def flaky_sha256_file(path: Path) -> str:
        if path.name == "bad.md":
            raise FileNotFoundError(path)  # as if deleted after Walk
        return real_sha256_file(path)

    monkeypatch.setattr(pipeline_mod, "sha256_file", flaky_sha256_file)

    # Walk (stat-only) then run the parse stage directly so the context's errors/parsed are
    # observable; this exercises the exact ``work()`` body where ``sha256_file`` is now guarded.
    pipe = _pipeline(jobs=jobs)
    ctx = pipe._new_context(src)
    pipe.stages()[0].run(ctx)  # WalkStage
    ctx = pipe.stages()[1].run(ctx)  # ResumableParseStage — must not raise

    skips = [e for e in ctx.errors if e.kind == "skip"]
    assert [e.item for e in skips] == ["bad.md"]
    assert skips[0].stage == "parse"
    assert "bad.md" in skips[0].message
    # The good file was still parsed; the bad one was not.
    parsed_paths = [ctx.space.document(doc_id).path for doc_id in ctx.parsed]  # type: ignore[union-attr]
    assert parsed_paths == ["good.md"]


# ── resume / cache ──────────────────────────────────────────────────────────


def test_resume_creates_cache_and_reproduces_output(tmp_path: Path) -> None:
    src = _tree(tmp_path / "src")
    out = tmp_path / "out"
    first = _pipeline(out=out, resume=True).run(src)
    assert (out / CACHE_DIRNAME).is_dir()
    # A second resumed run over unchanged input is byte-identical.
    second = _pipeline(out=out, resume=True).run(src)
    assert [c.id for c in first.chunks] == [c.id for c in second.chunks]
    assert [c.embedding for c in first.chunks] == [c.embedding for c in second.chunks]


def test_resume_reuses_parse_cache(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "doc.md").write_text("# Doc\n\nCached body text.\n")
    out = tmp_path / "out"

    counts = {"n": 0}

    class CountingParser:
        name = "counting"

        def __init__(self) -> None:
            from indx.parsers.plaintext import PlainTextParser

            self._inner = PlainTextParser()

        def parse(self, path: Path) -> object:
            counts["n"] += 1
            return self._inner.parse(path)

    pipe1 = DirectoryPipeline(**OFFLINE, out=out, resume=True)  # type: ignore[arg-type]
    pipe1.stages()[1]._parser = CountingParser()  # type: ignore[attr-defined]
    pipe1.run(src)
    assert counts["n"] == 1

    pipe2 = DirectoryPipeline(**OFFLINE, out=out, resume=True)  # type: ignore[arg-type]
    pipe2.stages()[1]._parser = CountingParser()  # type: ignore[attr-defined]
    pipe2.run(src)
    # The single file's parse result was served from cache, so the parser was not called.
    assert counts["n"] == 1


def test_resume_invalidates_on_parser_version_change(tmp_path: Path) -> None:
    # A parser whose registry name is unchanged but whose ``version`` bumps must NOT serve the
    # stale cached ParsedDoc under --resume: the version is folded into the cache component_id, so
    # the second pass is a clean cache miss and the changed parser is re-invoked (bug S2).
    src = tmp_path / "src"
    src.mkdir()
    (src / "doc.md").write_text("# Doc\n\nOriginal body text.\n")
    out = tmp_path / "out"

    counts = {"n": 0}

    class VersionedParser:
        name = "versioned"

        def __init__(self, version: str, marker: str) -> None:
            from indx.parsers.plaintext import PlainTextParser

            self.version = version
            self._marker = marker
            self._inner = PlainTextParser()

        def parse(self, path: Path) -> object:
            counts["n"] += 1
            doc = self._inner.parse(path)
            # Prepend a version-specific marker block so the resumed run's output is visibly v1 vs
            # v2 (text is a read-only property derived from blocks).
            for block in doc.blocks:
                block.order += 1
            doc.blocks.insert(0, Block(kind="text", text=self._marker, order=0))
            return doc

    pipe1 = DirectoryPipeline(  # type: ignore[arg-type]
        **{**OFFLINE, "parser": VersionedParser("1", "v1-marker")}, out=out, resume=True
    )
    space1 = pipe1.run(src)
    assert counts["n"] == 1  # parsed and cached
    assert any("v1-marker" in c.text for c in space1.chunks)

    pipe2 = DirectoryPipeline(  # type: ignore[arg-type]
        **{**OFFLINE, "parser": VersionedParser("2", "v2-marker")}, out=out, resume=True
    )
    space2 = pipe2.run(src)
    # The version bump invalidates the parse cache: the parser is re-invoked (cache miss) and the
    # produced chunks reflect the v2 output, not the stale v1 text.
    assert counts["n"] == 2
    assert any("v2-marker" in c.text for c in space2.chunks)
    assert not any("v1-marker" in c.text for c in space2.chunks)


def test_resume_off_writes_no_cache(tmp_path: Path) -> None:
    src = _tree(tmp_path / "src")
    out = tmp_path / "out"
    _pipeline(out=out, resume=False).run(src)
    assert not (out / CACHE_DIRNAME).exists()


# ── StageCache unit ─────────────────────────────────────────────────────────


def test_stage_cache_disabled_is_noop(tmp_path: Path) -> None:
    cache = StageCache(tmp_path, enabled=False)
    cache.put("parse", "h", "plaintext", {"x": 1})
    assert cache.get("parse", "h", "plaintext") is None
    assert not (tmp_path / CACHE_DIRNAME).exists()


def test_stage_cache_roundtrip_and_key_isolation(tmp_path: Path) -> None:
    cache = StageCache(tmp_path, enabled=True)
    cache.put("parse", "h1", "plaintext", {"v": [1.0, 2.0]})
    assert cache.get("parse", "h1", "plaintext") == {"v": [1.0, 2.0]}
    # Different component id / input hash / stage are isolated keys.
    assert cache.get("parse", "h1", "docling") is None
    assert cache.get("parse", "h2", "plaintext") is None
    assert cache.get("embed", "h1", "plaintext") is None


def test_sha256_text_is_stable() -> None:
    assert sha256_text("abc") == sha256_text("abc")
    assert sha256_text("abc") != sha256_text("abd")


def test_corrupt_cache_entry_is_a_miss(tmp_path: Path) -> None:
    cache = StageCache(tmp_path, enabled=True)
    cache.put("parse", "h", "plaintext", {"ok": True})
    # Corrupt the on-disk entry; get() must degrade to a miss, not raise.
    entry = next((tmp_path / CACHE_DIRNAME / "parse").glob("*.json"))
    entry.write_text("{not json")
    assert cache.get("parse", "h", "plaintext") is None


# ── build provenance honesty (Bug #5) ────────────────────────────────────────


def _enrich_stage(pipe: DirectoryPipeline) -> EnrichStage:
    stage = next(s for s in pipe.stages() if s.name == "enrich")
    assert isinstance(stage, EnrichStage)
    return stage


def test_build_does_not_claim_llm_or_vlm_provenance(tmp_path: Path) -> None:
    """build --llm/--vlm must not be recorded as build provenance — no LLM runs at build time.

    The deterministic Enrich stage uses no model, so stamping ``llm=openai`` into the
    manifest/components would be a false claim (Bug #5). Provenance for these two slots is ``none``
    regardless of what was requested; parser/embedder/store stay truthful.
    """
    pipe = DirectoryPipeline(
        parser="plaintext",
        store="jsonl",
        llm="openai",
        vlm="openai",
        embedder="hash",
        embed=False,
    )
    assert pipe._names["llm"] == "none"
    assert pipe._names["vlm"] == "none"
    # Regression guard: the fix only narrowed llm/vlm; other slots are still recorded truthfully.
    assert pipe._names["parser"] == "plaintext"
    assert pipe._names["store"] == "jsonl"


def test_manifest_components_report_llm_none_after_run(tmp_path: Path) -> None:
    src = _tree(tmp_path / "src")
    pipe = DirectoryPipeline(
        parser="plaintext", store="jsonl", embedder="hash", llm="openai", vlm="openai"
    )
    space = pipe.run(src)
    assert space.manifest.components["llm"] == "none"
    assert space.manifest.components["vlm"] == "none"
    assert space.manifest.components["parser"] == "plaintext"


def test_use_does_not_reintroduce_llm_provenance(tmp_path: Path) -> None:
    """Rebinding llm/vlm via use() must keep build provenance honest (none)."""
    pipe = _pipeline()
    pipe.use(llm="openai", vlm="openai")
    assert pipe._names["llm"] == "none"
    assert pipe._names["vlm"] == "none"


# ── [enrich] metadata config wired to the build stage (Bug #10) ───────────────


def test_enrich_metadata_config_selects_stage_fields(tmp_path: Path) -> None:
    """metadata=['summary'] config builds an EnrichStage wanting only summary (Bug #10)."""
    cfg = Config.model_validate({"enrich": {"metadata": ["summary"]}})
    pipe = DirectoryPipeline(config=cfg, parser="plaintext", store="jsonl", embed=False)
    stage = _enrich_stage(pipe)
    assert stage._want_summary is True
    assert stage._want_topics is False
    assert stage._want_tags is False
    assert stage._want_type is False


def test_enrich_metadata_subset_run_populates_only_summary(tmp_path: Path) -> None:
    cfg = Config.model_validate({"enrich": {"metadata": ["summary"]}})
    src = _tree(tmp_path / "src")
    pipe = DirectoryPipeline(config=cfg, parser="plaintext", store="jsonl", embedder="hash")
    space = pipe.run(src)
    doc = space.documents()[0]
    assert doc.summary is not None
    assert doc.topics == []
    assert doc.tags == []
    assert doc.doc_type is None


def test_enrich_metadata_default_populates_all_four(tmp_path: Path) -> None:
    """No metadata config → the build EnrichStage still populates all four fields (regression)."""
    pipe = DirectoryPipeline(parser="plaintext", store="jsonl", embed=False)
    stage = _enrich_stage(pipe)
    assert stage._want_summary is True
    assert stage._want_topics is True
    assert stage._want_tags is True
    assert stage._want_type is True

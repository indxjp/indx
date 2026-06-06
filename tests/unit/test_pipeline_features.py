"""Pipeline cross-cutting features: strict, dry-run/plan, jobs, resume.

These exercise the orchestration-layer behavior added to :class:`DirectoryPipeline` without
touching the individual stage files. Everything runs on the zero-dependency offline stack
(plaintext / hash / jsonl) so the suite stays network-free.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indx.core.context import SpaceContext, StageErrorRecord
from indx.errors import StageError
from indx.pipeline import BuildPlan, DirectoryPipeline
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

"""WalkFilter unit + WalkStage integration (Feature 5 — conditional / filtered import).

Pure offline-core-stack tests: no heavy extras, no network. The filter model is exercised
directly (parse_size, normalization, keep predicates) and through a real WalkStage over a temp
tree so the wiring, skip-count telemetry, and ``max_files`` determinism are covered end-to-end.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest
from pydantic import ValidationError

from indx.errors import ConfigError
from indx.pipeline import DirectoryPipeline, WalkFilter
from indx.pipeline.filters import parse_size

OFFLINE = {"parser": "plaintext", "llm": "none", "embedder": "hash", "store": "jsonl"}


# ── parse_size ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("512", 512),
        ("10kb", 10240),
        ("2mb", 2097152),
        ("1 gb", 1073741824),
        (1048576, 1048576),
        ("1.5kb", 1536),
    ],
)
def test_parse_size_ok(raw: int | str, expected: int) -> None:
    assert parse_size(raw) == expected


@pytest.mark.parametrize("bad", ["2gigs", "abc", "10 furlongs"])
def test_parse_size_bad_suffix_raises(bad: str) -> None:
    with pytest.raises(ConfigError):
        parse_size(bad)


def test_parse_size_negative_int_raises() -> None:
    with pytest.raises(ConfigError):
        parse_size(-5)


# ── normalization ────────────────────────────────────────────────────────────


def test_ext_normalization() -> None:
    f = WalkFilter(ext=["md", ".TXT", "  ", ""])
    assert f.ext == [".md", ".txt"]


def test_size_field_suffix_parsed_on_construction() -> None:
    f = WalkFilter(min_size="1kb", max_size="2mb")
    assert f.min_size == 1024
    assert f.max_size == 2097152


def test_negative_max_files_raises() -> None:
    with pytest.raises(ConfigError):
        WalkFilter(max_files=-1)


def test_negative_max_depth_raises() -> None:
    with pytest.raises(ConfigError):
        WalkFilter(max_depth=-2)


def test_unknown_field_forbidden() -> None:
    with pytest.raises(ValidationError):  # extra="forbid"
        WalkFilter(exclud=["x"])  # type: ignore[call-arg]


# ── is_empty ─────────────────────────────────────────────────────────────────


def test_is_empty_default_true() -> None:
    assert WalkFilter().is_empty() is True


@pytest.mark.parametrize(
    "kw",
    [
        {"include": ["a"]},
        {"exclude": ["a"]},
        {"ext": ["md"]},
        {"name": ["x"]},
        {"min_size": 1},
        {"max_size": 1},
        {"max_files": 1},
        {"max_depth": 0},
    ],
)
def test_is_empty_false_when_any_field_set(kw: dict[str, object]) -> None:
    assert WalkFilter(**kw).is_empty() is False  # type: ignore[arg-type]


# ── keep predicates ──────────────────────────────────────────────────────────


def _keep(f: WalkFilter, rel: str, *, size: int = 100, depth: int = 0) -> bool:
    return f.keep(PurePosixPath(rel), size_bytes=size, depth=depth)


def test_keep_ext_only() -> None:
    f = WalkFilter(ext=[".md"])
    assert _keep(f, "notes.md")
    assert not _keep(f, "notes.rst")


def test_keep_include_glob() -> None:
    f = WalkFilter(include=["src/**"])
    assert _keep(f, "src/a.md", depth=1)
    assert not _keep(f, "docs/a.md", depth=1)


def test_keep_include_leading_doublestar_matches_root() -> None:
    # a leading `**/` must match zero leading dirs (top-level) too, not only nested paths
    f = WalkFilter(include=["**/secret.md"])
    assert _keep(f, "secret.md", depth=0)  # top-level (was the regression case)
    assert _keep(f, "a/secret.md", depth=1)
    assert not _keep(f, "a/other.md", depth=1)


def test_keep_exclude_glob() -> None:
    f = WalkFilter(exclude=["**/_d/**"])
    assert _keep(f, "x/y.md", depth=1)
    assert not _keep(f, "x/_d/y.md", depth=2)
    assert not _keep(f, "_d/y.md", depth=1)  # top-level: leading `**/` collapses to zero dirs


def test_keep_exclude_leading_doublestar_drops_top_level_dir() -> None:
    # Regression for bug #1: README's `--exclude '**/_drafts/**'` must drop a TOP-LEVEL
    # _drafts/ dir, not only a nested one (fnmatch treated `/` as ordinary and silently leaked it).
    f = WalkFilter(exclude=["**/_drafts/**"])
    assert not _keep(f, "_drafts/secret.md", depth=1)  # depth=1 regression case
    assert not _keep(f, "sub/_drafts/secret.md", depth=2)  # nested still excluded
    assert _keep(f, "notes/a.md", depth=1)  # unrelated file kept


def test_keep_size_bounds_inclusive() -> None:
    f = WalkFilter(min_size=10, max_size=20)
    assert not _keep(f, "a.md", size=9)
    assert _keep(f, "a.md", size=10)  # lower bound inclusive
    assert _keep(f, "a.md", size=20)  # upper bound inclusive
    assert not _keep(f, "a.md", size=21)


def test_keep_name_glob_and_substring() -> None:
    f = WalkFilter(name=["README*"])
    assert _keep(f, "README.md")
    assert not _keep(f, "notes.md")
    g = WalkFilter(name=["spec"])
    assert _keep(g, "app.spec.md")  # substring match
    assert not _keep(g, "app.md")


def test_keep_max_depth() -> None:
    f = WalkFilter(max_depth=1)
    assert _keep(f, "a.md", depth=0)
    assert _keep(f, "d/a.md", depth=1)
    assert not _keep(f, "d/e/a.md", depth=2)


def test_keep_and_composition() -> None:
    f = WalkFilter(ext=[".md"], exclude=["**/draft/**"], max_size=100)
    assert _keep(f, "src/a.md", size=50, depth=1)
    assert not _keep(f, "src/a.txt", size=50, depth=1)  # wrong ext
    assert not _keep(f, "src/draft/a.md", size=50, depth=2)  # excluded
    assert not _keep(f, "src/a.md", size=200, depth=1)  # too big


# ── WalkStage integration over a real temp tree ──────────────────────────────


def _pipeline(flt: WalkFilter | None = None) -> DirectoryPipeline:
    return DirectoryPipeline(**OFFLINE, filter=flt)  # type: ignore[arg-type]


def _tree(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "a.md").write_text("alpha\n")
    (root / "b.txt").write_text("beta\n")
    (root / "c.rst").write_text("gamma\n")
    sub = root / "drafts"
    sub.mkdir()
    (sub / "d.md").write_text("delta\n")
    return root


def test_ext_filter_integration(tmp_path: Path) -> None:
    src = _tree(tmp_path / "src")
    space = _pipeline(WalkFilter(ext=[".md"])).run(src)
    paths = sorted(d.path for d in space.documents())
    assert paths == ["a.md", "drafts/d.md"]
    assert space.skipped_files_ == 2  # b.txt, c.rst


def test_exclude_subtree_integration(tmp_path: Path) -> None:
    src = _tree(tmp_path / "src")
    space = _pipeline(WalkFilter(exclude=["drafts/**"])).run(src)
    paths = [d.path for d in space.documents()]
    assert "drafts/d.md" not in paths
    assert "a.md" in paths
    assert space.skipped_files_ == 1


def test_exclude_top_level_dir_doublestar_integration(tmp_path: Path) -> None:
    # Regression for bug #1 over a real tree: the README pattern `**/_drafts/**` must drop a
    # TOP-LEVEL _drafts/ dir end-to-end (it leaked before the path-aware matcher).
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("alpha\n")
    drafts = src / "_drafts"
    drafts.mkdir()
    (drafts / "secret.md").write_text("ssh\n")
    space = _pipeline(WalkFilter(exclude=["**/_drafts/**"])).run(src)
    paths = [d.path for d in space.documents()]
    assert "_drafts/secret.md" not in paths
    assert "a.md" in paths
    assert space.skipped_files_ == 1


def test_max_size_drops_large_file(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "small.md").write_text("x")  # 1 byte
    (src / "big.md").write_text("y" * 3072)  # 3 KB
    space = _pipeline(WalkFilter(max_size="1kb")).run(src)
    paths = [d.path for d in space.documents()]
    assert paths == ["small.md"]
    assert space.skipped_files_ == 1


def test_max_size_boundary_is_raw_stat_bytes(tmp_path: Path) -> None:
    # Pins bug #S7 intent: size filtering is on the RAW on-disk stat byte count, inclusive.
    # A file of exactly 1024 bytes is kept at max_size='1kb'; 1025 bytes is dropped.
    src = tmp_path / "src"
    src.mkdir()
    (src / "exact.md").write_text("y" * 1024)  # exactly 1 KB on disk
    (src / "over.md").write_text("y" * 1025)  # one byte over
    space = _pipeline(WalkFilter(max_size="1kb")).run(src)
    paths = [d.path for d in space.documents()]
    assert paths == ["exact.md"]  # boundary inclusive on raw stat size
    assert space.skipped_files_ == 1


def test_max_files_caps_deterministically(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for n in ("e", "a", "c", "b", "d"):  # written out of order
        (src / f"{n}.md").write_text(n)
    space = _pipeline(WalkFilter(max_files=2)).run(src)
    paths = [d.path for d in space.documents()]
    assert paths == ["a.md", "b.md"]  # first two in sorted walk order
    assert space.skipped_files_ == 3
    # re-run yields an identical selection
    space2 = _pipeline(WalkFilter(max_files=2)).run(src)
    assert [d.path for d in space2.documents()] == paths


def test_max_files_uses_global_path_sort_not_dfs(tmp_path: Path) -> None:
    # DFS-with-per-directory-sort diverges from a global root-relative path sort when a root
    # file's stem prefixes a sibling directory name: '.' (0x2E) < '/' (0x2F), so ``sub.md``
    # sorts before ``sub/x.md`` globally, yet DFS descends into ``sub/`` first. The documented
    # cap is "first N in sorted walk order", so max_files=1 must keep ``sub.md``.
    src = tmp_path / "src"
    src.mkdir()
    (src / "sub").mkdir()
    (src / "sub" / "x.md").write_text("x")
    (src / "sub.md").write_text("s")
    space = _pipeline(WalkFilter(max_files=1)).run(src)
    paths = [d.path for d in space.documents()]
    assert paths == ["sub.md"]  # global-sort first, NOT the DFS-first 'sub/x.md'
    assert space.skipped_files_ == 1
    # selection is stable across re-runs
    space2 = _pipeline(WalkFilter(max_files=1)).run(src)
    assert [d.path for d in space2.documents()] == paths


def test_max_files_global_sort_with_dir_prefix_sibling(tmp_path: Path) -> None:
    # Same boundary with extra siblings: 'a.md' < 'a/b.md' globally, so it wins the cap.
    src = tmp_path / "src"
    src.mkdir()
    (src / "a").mkdir()
    (src / "a" / "b.md").write_text("b")
    (src / "a.md").write_text("a")
    (src / "z.md").write_text("z")
    space = _pipeline(WalkFilter(max_files=1)).run(src)
    assert [d.path for d in space.documents()] == ["a.md"]
    assert space.skipped_files_ == 2


def test_max_depth_zero_root_only(tmp_path: Path) -> None:
    src = _tree(tmp_path / "src")
    space = _pipeline(WalkFilter(max_depth=0)).run(src)
    paths = sorted(d.path for d in space.documents())
    assert paths == ["a.md", "b.txt", "c.rst"]
    assert all("/" not in p for p in paths)


def test_filters_compose_and(tmp_path: Path) -> None:
    src = _tree(tmp_path / "src")
    space = _pipeline(WalkFilter(ext=[".md"], max_files=1)).run(src)
    paths = [d.path for d in space.documents()]
    assert paths == ["a.md"]  # only the first .md in sorted order


def test_empty_selection_is_clean_no_op(tmp_path: Path) -> None:
    src = _tree(tmp_path / "src")
    out = tmp_path / "out"
    space = _pipeline(WalkFilter(ext=[".zzz"])).run(src, out)
    assert space.documents() == []
    assert space.skipped_files_ == 4
    # the writer still seals a valid (empty) archive
    assert (out / "handbook.indx").exists()


def test_no_filter_matches_unfiltered_walk(tmp_path: Path) -> None:
    src = _tree(tmp_path / "src")
    baseline = sorted(d.path for d in _pipeline(None).run(src).documents())
    filtered = sorted(d.path for d in _pipeline(WalkFilter()).run(src).documents())
    assert baseline == filtered == ["a.md", "b.txt", "c.rst", "drafts/d.md"]
    # skip count stays zero on the no-op path
    assert _pipeline(WalkFilter()).run(src).skipped_files_ == 0


# ── plan() reflects the filter ───────────────────────────────────────────────


def test_plan_reflects_filter(tmp_path: Path) -> None:
    src = _tree(tmp_path / "src")
    plan = _pipeline(WalkFilter(ext=[".md"])).plan(src)
    paths = sorted(d.path for d in plan.documents)
    assert paths == ["a.md", "drafts/d.md"]
    assert plan.skipped == 2


def test_plan_folders_reflect_kept_files(tmp_path: Path) -> None:
    src = _tree(tmp_path / "src")
    # exclude the drafts subtree → its folder should not appear
    plan = _pipeline(WalkFilter(exclude=["drafts/**"])).plan(src)
    assert "drafts" not in plan.folders

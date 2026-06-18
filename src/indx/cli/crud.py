"""`indx add` / `indx rm` / `indx update` — incremental CRUD over an existing .indx (Feature 3).

Thin CLI handles over the :class:`~indx.core.knowledge_space.KnowledgeSpace` mutators
(``add``/``remove``/``update``), preserving CLI⇄SDK parity. ``<space>`` accepts a ``.indx`` file
or an output directory (reuse :func:`resolve_archive` / :func:`load_space`); the archive is
resealed in place via :meth:`KnowledgeSpace.save` so the reseal is byte-deterministic.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import typer
from rich.markup import escape

from indx.cli._render import console, resolve_archive, resolve_target
from indx.core.knowledge_space import KnowledgeSpace
from indx.errors import ArchiveError


def _reseal_target(space_path: Path) -> Path:
    """Resolve the ``.indx`` archive a CRUD command reseals to (Feature 3).

    A ``.indx`` file reseals in place; an output dir reseals the contained ``*.indx``. A jsonl
    output dir (no ``.indx``) has nothing to reseal → :class:`ArchiveError` (exit 4), so the
    failure is typed rather than a crash.
    """
    path = Path(space_path)
    if path.is_dir() and not ((path / "handbook.indx").is_file() or any(path.glob("*.indx"))):
        raise ArchiveError(
            "CRUD reseals to .indx; this output directory holds a jsonl space "
            f"(no .indx archive found at {path})"
        )
    return resolve_archive(path)


def _save_target(space: Path | None, loaded: KnowledgeSpace) -> str:
    """Reseal ``loaded`` to its sink and return the human archive label (Feature 4).

    An explicit ``space`` reseals to its ``.indx`` archive; ``None`` reseals the home space via
    :func:`indx.home.save_home` and labels the home archive path.
    """
    if space is None:
        from indx.home import home_space_path, save_home

        save_home(loaded)
        return str(home_space_path())
    archive = _reseal_target(space)
    loaded.save(str(archive))
    return str(archive)


def _resolves_under_root(path: Path, source_root: str) -> bool:
    """True if ``path`` resolves under the space's ``source_root`` (mirrors _resolve_add_target).

    Uses the same candidate resolution as :meth:`KnowledgeSpace._resolve_add_target`: an absolute
    target is checked as-is; a relative one is tried against the CWD and against the source root
    (the SDK convention). When it falls under the root, the in-root ingest succeeds and needs no
    staging; otherwise the caller stages the external target instead of failing (#14 N5).
    """
    root = Path(source_root)
    if not root.is_dir():
        return False
    root_resolved = root.resolve()
    target = Path(path)
    candidates = [target] if target.is_absolute() else [Path.cwd() / target, root / target]
    for cand in candidates:
        try:
            cand.resolve().relative_to(root_resolved)
            return True
        except ValueError:
            continue
    return False


@contextmanager
def _ingest_arg(space: Path | None, loaded: KnowledgeSpace, path: Path) -> Iterator[str]:
    """Yield the ingest argument for ``add``/``update``, staging out-of-root targets (Feature 4).

    A target already **under** the space's ``manifest.source_root`` ingests in place (Feature 3,
    unchanged — no rebind, no staging — so its lineage/relative path and the deterministic reseal
    are preserved exactly). But a target from *anywhere else* must still be ingestible: the home
    space's ``source_root`` (``$INDX_HOME``) is never where documents live, and the README markets
    ``indx add ./notes/standup.md some.indx`` as general incremental ingestion (#14 N5). Pointing
    the ingest walk at an arbitrary parent (e.g. ``/tmp``) would crawl unrelated files, so an
    out-of-root target is rebound onto an isolated walk root:

    * a **file** is copied into a fresh temp dir whose only member is that file, so the ingest
      walks just it and ``Document.path`` is the bare basename;
    * a **directory** is used directly as the root (its tree is the intended corpus).

    The space's ``manifest.source_root`` is rebound to the chosen root only for the duration of the
    mutation, then restored to its original value, so the resealed manifest never persists a
    transient walk root — in particular not a temp dir that no longer exists. The temp dir (if any)
    is removed on exit.
    """
    # An explicit archive ingests an in-root target verbatim (the original Feature 3 path); the
    # home space (space is None) always stages, since its source_root is not the document tree.
    if space is not None and _resolves_under_root(path, loaded.manifest.source_root):
        yield str(path)
        return

    original_source_root = loaded.manifest.source_root
    abs_path = Path(path).resolve()
    if abs_path.is_dir():
        try:
            loaded.manifest.source_root = str(abs_path)
            yield "."
        finally:
            loaded.manifest.source_root = original_source_root
        return

    staged = Path(tempfile.mkdtemp(prefix="indx-add-"))
    try:
        shutil.copy2(abs_path, staged / abs_path.name)
        loaded.manifest.source_root = str(staged)
        yield abs_path.name
    finally:
        loaded.manifest.source_root = original_source_root
        shutil.rmtree(staged, ignore_errors=True)


def add_command(
    space: Path | None, path: Path, *, name: str | None = None, json_out: bool = False
) -> None:
    """Ingest ``path`` into ``space`` (or the home space) and reseal (``indx add``)."""
    # Validate an explicit sink up front so a jsonl-dir target fails before any ingest.
    if space is not None:
        _reseal_target(space)
    loaded = resolve_target(space)
    before_docs, before_chunks = len(loaded.documents_), len(loaded.chunks)
    with _ingest_arg(space, loaded, path) as arg:
        changed = loaded.add(arg)
    archive = _save_target(space, loaded)
    added_docs = len(loaded.documents_) - before_docs
    added_chunks = len(loaded.chunks) - before_chunks

    if json_out:
        payload = {
            "archive": archive,
            "changed": changed,
            "added": {"docs": added_docs, "chunks": added_chunks},
            "parse_failures": loaded.parse_failures_,
            "markup_paths": loaded.markup_paths_,
        }
        console.print_json(json.dumps(payload))
        return
    # L3: a no-op add (re-adding an unchanged/already-indexed/empty file) is legitimately
    # idempotent but a silent green ✓ also masks a typo'd path — AND a re-add can return a
    # non-empty `changed` list while contributing zero real deltas (+0 docs / +0 chunks), which
    # a green "added N doc(s) (+0/+0)" would misreport as success. Route to the yellow warning
    # whenever there are no real deltas (added_docs <= 0 and added_chunks <= 0), regardless of
    # `changed`, so a no-op reads as intended; a genuine add (positive deltas) still shows green.
    # Either way keep exit 0.
    if added_docs <= 0 and added_chunks <= 0:
        console.print(
            f"[yellow]![/yellow] no documents added (already indexed or empty?) → "
            f"[bold]{escape(archive)}[/bold]"
        )
        _warn_parse_failures(loaded.parse_failures_)
        _warn_markup_docs(loaded.markup_paths_)
        return
    console.print(
        f"[green]✓[/green] added {len(changed)} doc(s) "
        f"({added_docs:+d} docs · {added_chunks:+d} chunks) → "
        f"[bold]{escape(archive)}[/bold]"
    )
    # H1: surface any files that reached the parse stage but yielded 0 chunks.
    _warn_parse_failures(loaded.parse_failures_)
    # N6: surface any files indexed as raw structured markup rather than prose.
    _warn_markup_docs(loaded.markup_paths_)


def _warn_parse_failures(parse_failures: int) -> None:
    """Warn (yellow, exit 0) when ``parse_failures`` files were indexed with 0 chunks (H1)."""
    if parse_failures > 0:
        console.print(
            f"[yellow]![/yellow] {parse_failures} file(s) could not be parsed and were "
            "indexed with 0 chunks (try a richer parser, e.g. indx[docling])"
        )


def _warn_markup_docs(markup_paths: list[str]) -> None:
    """Warn (yellow, exit 0) when files were indexed as raw structured markup, not prose (N6)."""
    if markup_paths:
        shown = ", ".join(markup_paths[:3]) + (" …" if len(markup_paths) > 3 else "")
        console.print(
            f"[yellow]![/yellow] {len(markup_paths)} file(s) look like structured markup/data, "
            f"not prose, and were indexed as raw markup ({escape(shown)}) — retrieval quality "
            "may suffer; consider a format-aware parser or excluding them"
        )


def update_command(space: Path | None, path: Path, *, json_out: bool = False) -> None:
    """Re-ingest a changed ``path`` in ``space`` (or home) and reseal (``indx update``)."""
    if space is not None:
        _reseal_target(space)
    loaded = resolve_target(space)
    before_docs, before_chunks = len(loaded.documents_), len(loaded.chunks)
    with _ingest_arg(space, loaded, path) as arg:
        changed = loaded.update(arg)
    archive = _save_target(space, loaded)
    delta_docs = len(loaded.documents_) - before_docs
    delta_chunks = len(loaded.chunks) - before_chunks

    if json_out:
        payload = {
            "archive": archive,
            "changed": changed,
            "added": {"docs": delta_docs, "chunks": delta_chunks},
        }
        console.print_json(json.dumps(payload))
        return
    # A 0-change update is legitimate (the file was unchanged since the last index) but can also
    # mean a typo'd path; flag it distinctly, keeping exit 0.
    if not changed:
        console.print(
            f"[yellow]![/yellow] no documents changed (unchanged or not found?) → "
            f"[bold]{escape(archive)}[/bold]"
        )
        return
    console.print(
        f"[green]✓[/green] updated {len(changed)} doc(s) "
        f"({delta_docs:+d} docs · {delta_chunks:+d} chunks) → "
        f"[bold]{escape(archive)}[/bold]"
    )


def remove_command(space: Path | None, target: str, *, json_out: bool = False) -> None:
    """Drop a document (by id or path) from ``space`` (or home) and reseal (``indx rm``)."""
    if space is not None:
        _reseal_target(space)
    loaded = resolve_target(space)
    before_docs, before_chunks = len(loaded.documents_), len(loaded.chunks)
    removed = loaded.remove(target)
    archive = _save_target(space, loaded)
    removed_docs = before_docs - len(loaded.documents_)
    removed_chunks = before_chunks - len(loaded.chunks)

    if json_out:
        payload = {
            "archive": archive,
            "removed": {"ids": removed, "docs": removed_docs, "chunks": removed_chunks},
        }
        console.print_json(json.dumps(payload))
        return
    # Unlike add/update, a 0-match rm almost always means a wrong id/path — it can never be a
    # legitimate no-op the way re-adding an unchanged file is. Warn AND exit non-zero so a typo is
    # not a silent success (the JSON branch above is untouched, so scripted consumers are stable).
    if not removed:
        console.print(
            f"[yellow]![/yellow] no documents matched {escape(target)} — nothing removed from "
            f"[bold]{escape(archive)}[/bold]"
        )
        raise typer.Exit(1)
    console.print(
        f"[green]✓[/green] removed {len(removed)} doc(s) "
        f"({removed_docs} docs · {removed_chunks} chunks) → "
        f"[bold]{escape(archive)}[/bold]"
    )

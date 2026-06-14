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


@contextmanager
def _ingest_arg(space: Path | None, loaded: KnowledgeSpace, path: Path) -> Iterator[str]:
    """Yield the ingest argument for ``add``/``update``, staging home targets (Feature 4).

    An explicit space ingests ``path`` under its own ``manifest.source_root`` (Feature 3,
    unchanged — no rebind, no staging). The home space, however, must accept a file/dir from
    *anywhere*: its ``source_root`` (``$INDX_HOME``) is not where a user's documents live, and
    pointing the ingest walk at an arbitrary parent (e.g. ``/tmp``) would crawl unrelated files.
    So a home target is rebound onto an isolated walk root:

    * a **file** is copied into a fresh temp dir whose only member is that file, so the ingest
      walks just it and ``Document.path`` is the bare basename;
    * a **directory** is used directly as the root (its tree is the intended corpus).

    The space's ``manifest.source_root`` is rebound to the chosen root only for the duration of the
    mutation, then restored to its original value (``$INDX_HOME``) so the resealed home manifest
    never persists a transient walk root — in particular not a temp dir that no longer exists. The
    temp dir (if any) is removed on exit.
    """
    if space is not None:
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

    staged = Path(tempfile.mkdtemp(prefix="indx-home-add-"))
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
        }
        console.print_json(json.dumps(payload))
        return
    console.print(
        f"[green]✓[/green] added {len(changed)} doc(s) "
        f"({added_docs:+d} docs · {added_chunks:+d} chunks) → "
        f"[bold]{escape(archive)}[/bold]"
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
    console.print(
        f"[green]✓[/green] removed {len(removed)} doc(s) "
        f"({removed_docs} docs · {removed_chunks} chunks) → "
        f"[bold]{escape(archive)}[/bold]"
    )

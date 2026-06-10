"""ZIP input support — extract a ``.zip`` build input into a temp directory, safely.

``indx <archive.zip> --out …`` and ``DirectoryPipeline().run("corpus.zip", …)`` accept a ZIP
in place of a directory (pipeline/walk.md). The archive is extracted into a fresh temp
directory which then becomes the walk ``root``; the pipeline is otherwise unchanged. The
temp directory is the pipeline's responsibility to remove once the run (or dry-run) ends.

Extraction is **zip-slip safe**: every member is resolved against the extraction root and
any entry that would write or resolve outside it (absolute paths or ``../`` traversal) is
rejected. Symlink members are rejected outright (they are never materialized as symlinks),
so an in-archive link can never redirect a later member's write outside the tree
(walk.md §"Security & robustness").
"""

from __future__ import annotations

import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

from indx.errors import IndxError

_TEMP_PREFIX = "indx-zip-"

# Cap on total *uncompressed* extracted size. DEFLATE can expand ~1000x, so a tiny archive can
# decompress to many GB; without a cap a small upload becomes an unauthenticated disk-fill DoS
# (POST /api/import → build → extract_zip). 2 GiB comfortably covers any real build corpus.
_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024


class ZipInputError(IndxError):
    """A build input ``.zip`` is unreadable or contains an unsafe (zip-slip) member."""


def is_zip_input(path: Path) -> bool:
    """True if ``path`` is a file indx should treat as a ZIP build input."""
    return path.is_file() and path.suffix.lower() == ".zip"


def extract_zip(zip_path: Path) -> Path:
    """Extract ``zip_path`` into a new temp directory and return that directory.

    The caller owns the returned directory and must remove it when done. Raises
    :class:`ZipInputError` on a corrupt archive, any member that escapes the root, or any
    symlink member.
    """
    zip_path = Path(zip_path)
    dest = Path(tempfile.mkdtemp(prefix=_TEMP_PREFIX))
    dest_resolved = dest.resolve()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            # Pre-check the declared uncompressed total before writing anything (cheap, header-only)
            # so a zip bomb is rejected before it fills the disk.
            total = sum(info.file_size for info in zf.infolist())
            if total > _MAX_TOTAL_BYTES:
                raise ZipInputError(
                    f"refusing oversized zip input: {total} uncompressed bytes "
                    f"exceeds {_MAX_TOTAL_BYTES} in {zip_path}"
                )
            for info in zf.infolist():
                member = info.filename
                # Reject symlink members outright. We never materialize them as symlinks, but
                # an explicit check keeps the contract real and stops a future symlink-preserving
                # change from redirecting a later member's write past the zip-slip check below.
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ZipInputError(
                        f"refusing unsafe zip member (symlink): {member!r} in {zip_path}"
                    )
                # Reject any member whose resolved target leaves the extraction root.
                target = (dest / member).resolve()
                if target != dest_resolved and dest_resolved not in target.parents:
                    raise ZipInputError(
                        f"refusing unsafe zip member (zip-slip): {member!r} in {zip_path}"
                    )
            # Stream each member into dest with a running budget so a header that lies about
            # file_size cannot exceed the cap at write time either.
            written = 0
            for info in zf.infolist():
                if info.is_dir():
                    (dest / info.filename).mkdir(parents=True, exist_ok=True)
                    continue
                target = dest / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out_fh:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > _MAX_TOTAL_BYTES:
                            raise ZipInputError(
                                f"refusing oversized zip input: extracted bytes exceed "
                                f"{_MAX_TOTAL_BYTES} in {zip_path}"
                            )
                        out_fh.write(chunk)
    except ZipInputError:
        _cleanup(dest)
        raise
    except (zipfile.BadZipFile, OSError) as exc:
        _cleanup(dest)
        raise ZipInputError(f"could not read zip input {zip_path}: {exc}") from exc
    return dest


def _cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def cleanup_extracted(path: Path | None) -> None:
    """Remove a temp directory created by :func:`extract_zip` (no-op if ``None``)."""
    if path is not None:
        _cleanup(path)

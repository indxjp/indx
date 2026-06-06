"""ZIP input support — extract a ``.zip`` build input into a temp directory, safely.

``indx <archive.zip> --out …`` and ``DirectoryPipeline().run("corpus.zip", …)`` accept a ZIP
in place of a directory (pipeline/walk.md). The archive is extracted into a fresh temp
directory which then becomes the walk ``root``; the pipeline is otherwise unchanged. The
temp directory is the pipeline's responsibility to remove once the run (or dry-run) ends.

Extraction is **zip-slip safe**: every member is resolved against the extraction root and
any entry that would write or resolve outside it (absolute paths, ``../`` traversal, or a
symlink escaping the tree) is rejected (walk.md §"Security & robustness").
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from indx.errors import IndxError

_TEMP_PREFIX = "indx-zip-"


class ZipInputError(IndxError):
    """A build input ``.zip`` is unreadable or contains an unsafe (zip-slip) member."""


def is_zip_input(path: Path) -> bool:
    """True if ``path`` is a file indx should treat as a ZIP build input."""
    return path.is_file() and path.suffix.lower() == ".zip"


def extract_zip(zip_path: Path) -> Path:
    """Extract ``zip_path`` into a new temp directory and return that directory.

    The caller owns the returned directory and must remove it when done. Raises
    :class:`ZipInputError` on a corrupt archive or any member that escapes the root.
    """
    zip_path = Path(zip_path)
    dest = Path(tempfile.mkdtemp(prefix=_TEMP_PREFIX))
    dest_resolved = dest.resolve()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                # Reject any member whose resolved target leaves the extraction root.
                target = (dest / member).resolve()
                if target != dest_resolved and dest_resolved not in target.parents:
                    raise ZipInputError(
                        f"refusing unsafe zip member (zip-slip): {member!r} in {zip_path}"
                    )
            zf.extractall(dest)
    except ZipInputError:
        _cleanup(dest)
        raise
    except (zipfile.BadZipFile, OSError) as exc:
        _cleanup(dest)
        raise ZipInputError(f"could not read zip input {zip_path}: {exc}") from exc
    return dest


def _cleanup(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def cleanup_extracted(path: Path | None) -> None:
    """Remove a temp directory created by :func:`extract_zip` (no-op if ``None``)."""
    if path is not None:
        _cleanup(path)

"""WalkFilter — declarative, deterministic file selection for the Walk stage (Feature 5).

Pure-stdlib predicate model: no heavy deps, works on the offline core stack. The filter is
applied inside WalkStage against each candidate's root-relative path + lineage + stat; it never
mutates iter_files (which stays generic). A file is kept iff it passes ALL *set* predicates.
"""

from __future__ import annotations

import fnmatch
import re
from functools import lru_cache
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator

from indx.errors import ConfigError

# kb/mb/gb (and bare bytes / "b") — case-insensitive, optional space, decimal allowed.
_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmg]?b?)\s*$", re.IGNORECASE)
_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
}


@lru_cache(maxsize=512)
def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile a path-aware glob to an anchored regex (cached per distinct pattern).

    Unlike :func:`fnmatch.fnmatch`, ``/`` is significant here so ``**`` semantics work:
    a leading ``**/`` matches zero-or-more leading dirs (so ``**/_drafts/**`` excludes a
    *top-level* ``_drafts/`` as well as a nested one), bare ``**`` spans separators, while
    ``*``/``?`` stay within a single path segment. All other characters are matched literally.
    """
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                # `**` — consume an optional trailing `/` so `**/` collapses to zero dirs.
                i += 2
                if i < n and pattern[i] == "/":
                    out.append("(?:.*/)?")
                    i += 1
                else:
                    out.append(".*")
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile(rf"(?s:{''.join(out)})\Z")


def _path_match(rel_str: str, pattern: str) -> bool:
    """Path-aware glob match against a root-relative POSIX path (see :func:`_glob_to_regex`)."""
    return _glob_to_regex(pattern).match(rel_str) is not None


def parse_size(value: int | str) -> int:
    """Parse a human size (``"10kb"``, ``"2 mb"``, ``"512"``, ``1048576``) to a byte count.

    Bare ints/numeric strings are bytes. Suffixes are binary (kb=1024). Raises ConfigError
    (CLI exit 3) on an unparseable token so a bad ``[walk] max_size`` / ``--max-size`` fails
    fast with an actionable message instead of silently selecting everything.
    """
    if isinstance(value, int):
        if value < 0:
            raise ConfigError(f"size must be non-negative: {value}")
        return value
    m = _SIZE_RE.match(value)
    if not m:
        raise ConfigError(
            f"invalid size {value!r}: use bytes or a kb/mb/gb suffix (e.g. '10kb', '2mb')"
        )
    return int(float(m.group(1)) * _SIZE_UNITS[m.group(2).lower()])


class WalkFilter(BaseModel):
    """Declarative selection of files entering the space at build time (Feature 5).

    Every field is optional; an unset field imposes no constraint. A file is kept iff it
    passes ALL set predicates (logical AND). Predicates run on the *root-relative POSIX path*
    and the file ``stat`` — never absolute paths — so selection is portable and deterministic.
    """

    model_config = ConfigDict(extra="forbid")

    include: list[str] = Field(default_factory=list)  # `**`-aware path globs; keep iff matches ANY
    exclude: list[str] = Field(default_factory=list)  # `**`-aware path globs; drop iff matches ANY
    ext: list[str] = Field(default_factory=list)  # extensions, normalized to leading-dot lower
    name: list[str] = Field(default_factory=list)  # filename glob OR substring; keep iff ANY
    min_size: int | None = None  # bytes (raw stat size); inclusive lower bound
    max_size: int | None = None  # bytes (raw stat size); inclusive upper bound
    max_files: int | None = None  # cap; first-N after the walk's stable sort
    max_depth: int | None = None  # lineage depth (len(lineage)); 0 = root only

    # ---- normalization (validators run on construction, incl. from-config and from-CLI) ----

    @field_validator("ext", mode="after")
    @classmethod
    def _norm_ext(cls, raw: list[str]) -> list[str]:
        out: list[str] = []
        for e in raw:
            e = e.strip().lower()
            if not e:
                continue
            out.append(e if e.startswith(".") else f".{e}")
        return out

    @field_validator("min_size", "max_size", mode="before")
    @classmethod
    def _parse_size_field(cls, v: int | str | None) -> int | None:
        return None if v is None else parse_size(v)

    @field_validator("max_files", "max_depth", mode="after")
    @classmethod
    def _non_negative(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ConfigError(f"value must be non-negative, got {v}")
        return v

    # ---- predicates ----

    def is_empty(self) -> bool:
        """True when no predicate is set (the walk yields everything; a fast no-op path)."""
        return not (
            self.include
            or self.exclude
            or self.ext
            or self.name
            or self.min_size is not None
            or self.max_size is not None
            or self.max_files is not None
            or self.max_depth is not None
        )

    def keep(self, rel: PurePosixPath, *, size_bytes: int, depth: int) -> bool:
        """Decide a single candidate (max_files is applied by the caller, post-sort).

        ``rel`` is the root-relative POSIX path; ``size_bytes`` is the raw on-disk ``stat``
        size (the walk runs pre-parse); ``depth`` is ``len(lineage)`` (the number of folders
        between root and the file). All checks are AND-combined; ``max_files`` is NOT checked
        here — it is a stateful cap the stage applies after the deterministic sort.
        """
        rel_str = rel.as_posix()
        name = rel.name
        if self.max_depth is not None and depth > self.max_depth:
            return False
        if self.ext and rel.suffix.lower() not in self.ext:
            return False
        if self.min_size is not None and size_bytes < self.min_size:
            return False
        if self.max_size is not None and size_bytes > self.max_size:
            return False
        if self.include and not any(_path_match(rel_str, g) for g in self.include):
            return False
        if self.exclude and any(_path_match(rel_str, g) for g in self.exclude):
            return False
        return not (self.name and not any(fnmatch.fnmatch(name, g) or g in name for g in self.name))

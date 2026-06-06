"""Cross-cutting helpers. No domain logic; imports nothing internal beyond errors."""

from indx.utils.hashing import stable_hash
from indx.utils.io import iter_files, read_text
from indx.utils.lazy import require_extra

__all__ = ["stable_hash", "iter_files", "read_text", "require_extra"]

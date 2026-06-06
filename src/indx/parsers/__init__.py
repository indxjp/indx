"""Parser slot. Default = docling (extra); plaintext ships in core as a zero-dep fallback.

Re-exports the ``Parser`` protocol (and ``ParsedDoc``) so ``from indx.parsers import Parser``
works as the SDK surface documents (technical-spec §8.1).
"""

from indx.core.parsed import ParsedDoc
from indx.parsers.base import Parser

__all__ = ["ParsedDoc", "Parser"]

"""A minimal third-party-style parser plugin for indx.

This is a *teaching example* of the "add a backend without forking" pattern. It implements
the parser slot's structural ``typing.Protocol`` (``indx.parsers.base.Parser``) — note there
is no base class to subclass: the registry checks the shape, so any object with a ``name``
attribute and a ``parse(path) -> ParsedDoc`` method qualifies.

A real plugin would advertise this class under the ``indx.parsers`` entry-point group so the
registry discovers it after its first-party builtins (builtins always win name collisions).
See ``pyproject.toml`` in this directory for the entry-point declaration.

Lazy-import note: this parser is pure-Python and has no vendor SDK, so there is nothing to
gate. A backend that wraps a heavy/optional dependency must import that SDK *inside*
``__init__`` and call ``indx.utils.lazy.require_extra(...)`` first, so importing the module
stays cheap and a missing extra surfaces as a friendly ``MissingExtraError`` only when the
slot is actually selected.
"""

from __future__ import annotations

from pathlib import Path

from indx.core.parsed import Block, ParsedDoc


class UppercaseParser:
    """Reads a text file and emits a single uppercased text block.

    Structurally satisfies ``indx.parsers.base.Parser``: it exposes ``name`` and a
    ``parse(path) -> ParsedDoc`` method. Like every local parser, it must not touch the
    network.
    """

    name = "uppercase"
    version = "1"

    def parse(self, path: Path) -> ParsedDoc:
        text = path.read_text(encoding="utf-8")
        block = Block(kind="text", text=text.upper().strip(), order=0)
        return ParsedDoc(
            source_path=str(path),
            parser=self.name,
            parser_version=self.version,
            blocks=[block],
        )


# --- For a vendor-backed backend, the constructor would look like this -----------------
#
#     class MyVendorParser:
#         name = "myvendor"
#
#         def __init__(self) -> None:
#             # Gate the optional extra FIRST, then lazy-import the SDK so the module stays
#             # import-clean on a bare install. Signature is
#             # require_extra(slot, name, extra, *module_names):
#             from indx.utils.lazy import require_extra  # pragma: no cover - illustration
#
#             require_extra("parser", "myvendor", "myvendor", "myvendor_sdk")
#             import myvendor_sdk  # imported here, never at module top level
#
#             self._client = myvendor_sdk.Client()
#
#         def parse(self, path: Path) -> ParsedDoc: ...

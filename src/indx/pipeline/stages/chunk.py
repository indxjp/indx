"""03 Chunk — split ParsedDoc into Chunks with position + neighbor links (FR-S03)."""

from __future__ import annotations

from indx.core.chunk import Chunk
from indx.core.context import SpaceContext
from indx.utils.hashing import stable_hash


class ChunkStage:
    name = "chunk"

    def __init__(self, max_chars: int = 800) -> None:
        self.max_chars = max_chars

    def run(self, ctx: SpaceContext) -> SpaceContext:
        for doc in ctx.space.documents_:
            parsed = ctx.parsed.get(doc.id)
            if parsed is None:
                continue
            pieces = _pack(parsed.text, self.max_chars)
            made: list[Chunk] = []
            for pos, text in enumerate(pieces):
                made.append(
                    Chunk(id=stable_hash(f"{doc.id}:{pos}"), doc_id=doc.id, position=pos, text=text)
                )
            _link(made)  # prev/next within the document (in-doc `continues`)
            ctx.space.chunks.extend(made)
        return ctx


def _pack(text: str, max_chars: int) -> list[str]:
    """Greedy paragraph packing up to ``max_chars``; structure-aware, deterministic.

    Any packed piece still longer than ``max_chars`` (e.g. a single paragraph
    that exceeds the limit on its own) is hard-split on a deterministic
    character window so ``max_chars`` is an upper bound, not a soft target.
    """
    packed: list[str] = []
    buf = ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if buf and len(buf) + len(para) + 2 > max_chars:
            packed.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        packed.append(buf)
    packed = packed or ([text.strip()] if text.strip() else [])
    out: list[str] = []
    for piece in packed:
        if len(piece) <= max_chars:
            out.append(piece)
        else:
            out.extend(piece[i : i + max_chars] for i in range(0, len(piece), max_chars))
    return out


def _link(chunks: list[Chunk]) -> None:
    for i, c in enumerate(chunks):
        if i > 0:
            c.prev_id = chunks[i - 1].id
        if i < len(chunks) - 1:
            c.next_id = chunks[i + 1].id

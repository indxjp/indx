"""04 Relate — derive Relation edges across documents (FR-S04).

Deterministic, high-precision relations only — favoring precision over recall (PRD risk R-3):

* ``sibling`` (same folder), ``parent`` (folder index doc -> its folder's docs), and
  ``continues`` (numbered multi-part files in a folder) — the structural P0 relations.
* ``references`` — a document that mentions another document by filename, relative path, or
  markdown link. A mention is only emitted when it resolves to exactly one walked document
  (ambiguous or unresolved mentions are dropped).
* ``duplicate-of`` — near/exact duplicate documents detected by a normalized-text content hash
  (Relate runs before Embed, so this uses text, never vectors).

``references`` and ``duplicate-of`` read parsed text from ``ctx.parsed``; when a document has
no parsed entry (e.g. Relate is run standalone without Parse) it simply contributes no
content-derived edges.
"""

from __future__ import annotations

import hashlib
import logging
import re

from indx.core.context import SpaceContext
from indx.core.document import Document
from indx.core.relation import Relation, RelationType

# A numbered "part" suffix: a base name followed by an explicit ``part``/``p`` marker and a
# number (``guide-part2``, ``guide_p3``). The marker is required so bare numeric suffixes —
# years (``report2024``), versions (``v2``), quarters (``q1``) — are NOT mistaken for parts,
# preserving the module's high-precision contract.
_PART = re.compile(r"(.+?)(?:[ _-]*part[ _-]?|[ _-]+p)(\d+)\b", re.IGNORECASE)
# A stem carrying MORE THAN ONE part marker (``report-p1-p2``) is ambiguous: ``fullmatch``
# would anchor on the trailing marker and silently fold earlier ones into the base, bucketing
# it under a different sequence than its single-marker namesakes (``report-p1``/``report-p2``).
# We detect and skip such stems rather than group them wrongly (precision over recall, R-3).
_PART_MARKER = re.compile(r"(?:[ _-]*part[ _-]?|[ _-]+p)\d+\b", re.IGNORECASE)
_INDEX_STEMS = {"index", "readme", "_index", "overview"}

# A markdown inline link target: the URL/path inside ``](...)``. We only keep the path part,
# dropping any title or fragment/anchor, so ``[x](docs/a.md#sec "t")`` resolves to ``docs/a.md``.
_MD_LINK = re.compile(r"\]\(\s*<?([^)>\s]+)>?(?:\s+[^)]*)?\)")

# Whitespace runs collapse to a single space when normalizing text for duplicate detection.
_WS = re.compile(r"\s+")

# Maximum group size for pairwise SIBLING emission. A flat directory at or below this size gets
# the full symmetric pairing; ABOVE it, pairwise emission is O(n^2) (a 500-file folder is already
# ~125k edges), so we degrade gracefully to a bounded per-doc neighbourhood instead of either
# exploding or — as before — silently emitting ZERO sibling edges for the whole folder (#14 N4).
SIBLING_MAX = 500
# Above SIBLING_MAX, each doc links to its next k path-adjacent siblings (the group is path
# sorted, so this is deterministic). Bounds the edge count at ~k*n while keeping every large
# folder some local sibling structure rather than none.
SIBLING_NEIGHBORS = 8

_log = logging.getLogger(__name__)


class RelateStage:
    name = "relate"

    def run(self, ctx: SpaceContext) -> SpaceContext:
        docs = ctx.space.documents_
        with_content = _docs_with_content(ctx)
        rels: list[Relation] = []
        by_folder: dict[tuple[str, ...], list[Document]] = {}
        for d in docs:
            by_folder.setdefault(tuple(d.lineage), []).append(d)

        for group in by_folder.values():
            group = sorted(group, key=lambda d: d.path)
            # sibling: pair documents in the same folder, but exclude 0-chunk shells (binaries,
            # empty/whitespace files) — they are non-retrievable and pairing them is pure noise
            # (#14 N4). When content is unknown (Relate run standalone after Walk only), keep the
            # full group so the purely-structural relate path is unchanged.
            sib_group = (
                group if with_content is None else [d for d in group if d.id in with_content]
            )
            rels.extend(_siblings(sib_group))
            # parent: a folder index/readme is the parent of its folder-mates
            indexes = [d for d in group if _stem(d).lower() in _INDEX_STEMS]
            for idx in indexes:
                for other in group:
                    if other.id != idx.id:
                        rels.append(Relation(src=idx.id, dst=other.id, type=RelationType.PARENT))
            # continues: numbered parts that share a base name, in numeric order
            rels.extend(_continues(group))

        # Content-derived, cross-folder relations. These read parsed text, so they are emitted
        # only for documents Parse has populated in ctx.parsed.
        rels.extend(_references(ctx))
        rels.extend(_duplicates(ctx))

        ctx.space.relations.extend(rels)
        return ctx


def _docs_with_content(ctx: SpaceContext) -> set[str] | None:
    """Doc ids carrying indexable content, or ``None`` when content is not yet known.

    Returns ``None`` only when neither chunks nor parsed text are available — i.e. Relate was run
    standalone right after Walk (no Parse/Chunk). Callers then skip the 0-chunk sibling filter and
    keep the purely structural behavior. Otherwise:

    * after Chunk (the build path), a doc "has content" iff it owns ≥1 chunk;
    * before Chunk but after Parse (the relate-quality tests, and the reseal recompute that rebuilds
      ``ctx.parsed`` from chunk text), iff its parsed text is non-empty.
    """
    if ctx.space.chunks:
        return {c.doc_id for c in ctx.space.chunks}
    if ctx.parsed:
        return {doc_id for doc_id, p in ctx.parsed.items() if p is not None and p.text.strip()}
    return None


def _siblings(group: list[Document]) -> list[Relation]:
    """Sibling edges within one (path-sorted) folder group, bounded above ``SIBLING_MAX``.

    At or below the threshold, every unordered pair is linked (full symmetric pairing). Above it,
    pairwise emission would be O(n^2), so each doc links only to its next :data:`SIBLING_NEIGHBORS`
    path-adjacent siblings — deterministic, bounded at ~k*n, and (unlike the old hard cap) never
    zero for a large folder (#14 N4).
    """
    out: list[Relation] = []
    if len(group) <= SIBLING_MAX:
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                out.append(Relation(src=a.id, dst=b.id, type=RelationType.SIBLING))
    else:
        _log.info(
            "relate: folder with %d docs exceeds SIBLING_MAX=%d; emitting a bounded "
            "%d-neighbour sibling graph instead of full pairwise edges",
            len(group),
            SIBLING_MAX,
            SIBLING_NEIGHBORS,
        )
        for i, a in enumerate(group):
            for b in group[i + 1 : i + 1 + SIBLING_NEIGHBORS]:
                out.append(Relation(src=a.id, dst=b.id, type=RelationType.SIBLING))
    return out


def _stem(doc: Document) -> str:
    name = doc.path.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0]


def _continues(group: list[Document]) -> list[Relation]:
    parts: dict[str, list[tuple[int, Document]]] = {}
    for d in group:
        stem = _stem(d)
        # Ambiguous multi-marker stems (e.g. ``report-p1-p2``) are excluded — see _PART_MARKER.
        if len(_PART_MARKER.findall(stem)) > 1:
            continue
        m = _PART.fullmatch(stem)
        if m and m.group(1):
            parts.setdefault(m.group(1).rstrip(" _-").lower(), []).append((int(m.group(2)), d))
    out: list[Relation] = []
    for seq in parts.values():
        if len(seq) < 2:
            continue
        seq.sort(key=lambda t: t[0])
        for (_, a), (_, b) in zip(seq, seq[1:], strict=False):
            out.append(Relation(src=a.id, dst=b.id, type=RelationType.CONTINUES))
    return out


def _doc_text(ctx: SpaceContext, doc: Document) -> str | None:
    parsed = ctx.parsed.get(doc.id)
    return parsed.text if parsed is not None else None


# Source extensions whose link structure lives in the *raw* file as plain text. For these we scan
# the source on disk rather than the parser-normalized text: a normalizing parser (docling, the
# default) discards link targets — ``[Beta](beta.md)`` becomes the bare word ``Beta`` — so mention
# extraction from ``ctx.parsed`` finds nothing and ``references`` never fires (issue #17 bug 2).
# Binary formats (docx/pdf/odt) carry no textual link syntax, so they keep using parsed text.
_TEXTUAL_EXTS = frozenset(
    {
        ".md",
        ".markdown",
        ".mdx",
        ".txt",
        ".rst",
        ".org",
        ".html",
        ".htm",
        ".xhtml",
        ".tex",
        ".adoc",
        ".asciidoc",
        ".wiki",
    }
)


def _mention_source_text(ctx: SpaceContext, doc: Document) -> str | None:
    """Text to scan for outbound mentions, parser-independently.

    Returns ``None`` for a document Parse never populated, so the "no parse ⇒ no content-derived
    edges" contract of a standalone Relate is preserved. For a parsed textual document the raw file
    on disk is read instead — link targets survive there regardless of which parser ran, whereas a
    normalizing parser (docling, the default) drops them from the parsed text (issue #17 bug 2). If
    the raw read fails (tree absent, I/O error) it falls back to the parser-normalized text; a
    non-textual format always uses parsed text (it carries no textual link syntax).
    """
    parsed_text = _doc_text(ctx, doc)
    if parsed_text is None:
        return None
    ext = ("." + doc.path.rsplit(".", 1)[-1].lower()) if "." in doc.path else ""
    if ext in _TEXTUAL_EXTS:
        try:
            return (ctx.root / doc.path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            pass
    return parsed_text


def _norm_path(path: str) -> str:
    """Forward-slash, lowercased, ``./`` and trailing-slash stripped — for matching mentions."""
    p = path.replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    return p.rstrip("/").lower()


def _references(ctx: SpaceContext) -> list[Relation]:
    """Resolve in-text mentions of other documents to ``references`` edges (high precision).

    A mention resolves only when it maps to exactly one walked document; ambiguous basenames
    (the same filename in two folders) and unknown targets are dropped, and a document never
    references itself.
    """
    docs = ctx.space.documents_
    # Index walked documents by their full relative path and (separately) by bare basename.
    # A basename that occurs more than once is ambiguous and excluded from resolution.
    by_path: dict[str, Document] = {}
    by_base: dict[str, list[Document]] = {}
    for d in docs:
        by_path[_norm_path(d.path)] = d
        base = _norm_path(d.path).rsplit("/", 1)[-1]
        by_base.setdefault(base, []).append(d)

    out: list[Relation] = []
    seen: set[tuple[str, str]] = set()  # dedupe repeated mentions of the same target
    for src in docs:
        text = _mention_source_text(ctx, src)
        if not text:
            continue
        for token in _mention_tokens(text):
            target = _resolve_mention(token, by_path, by_base)
            if target is None or target.id == src.id:
                continue
            key = (src.id, target.id)
            if key in seen:
                continue
            seen.add(key)
            out.append(Relation(src=src.id, dst=target.id, type=RelationType.REFERENCES))
    return out


def _mention_tokens(text: str) -> list[str]:
    """Candidate path/filename mentions: markdown link targets plus bare path-like tokens."""
    tokens: list[str] = []
    tokens.extend(m.group(1) for m in _MD_LINK.finditer(text))
    # Bare tokens that look like a path or a file with an extension (e.g. ``guide.md`` or
    # ``docs/guide.md``). Requires a dot-extension to stay conservative (precision over recall).
    for raw in re.split(r"[\s,;:()\[\]{}<>\"'`|]+", text):
        tok = raw.strip().rstrip(".")
        if "." in tok.rsplit("/", 1)[-1] and not tok.startswith("http"):
            tokens.append(tok)
    return tokens


def _resolve_mention(
    token: str, by_path: dict[str, Document], by_base: dict[str, list[Document]]
) -> Document | None:
    """Map one mention token to a single document, or ``None`` if unresolved/ambiguous."""
    if token.startswith(("http://", "https://", "mailto:", "#")):
        return None
    # Drop a URL fragment/anchor or query so ``a.md#sec`` resolves to ``a.md``.
    cand = _norm_path(re.split(r"[#?]", token, maxsplit=1)[0])
    if not cand:
        return None
    if cand in by_path:
        return by_path[cand]
    base = cand.rsplit("/", 1)[-1]
    if "/" not in cand:
        # A bare basename resolves only when exactly one document carries it (unambiguous).
        matches = by_base.get(base, [])
        if len(matches) == 1:
            return matches[0]
        return None
    # A relative path that did not match a full path: try suffix-matching its basename, but only
    # accept it when that basename is itself globally unique (still high precision).
    matches = by_base.get(base, [])
    if len(matches) == 1:
        stored = _norm_path(matches[0].path)
        if stored == cand or stored.endswith("/" + cand):
            return matches[0]
    return None


def _normalize_for_hash(text: str) -> str:
    """Collapse whitespace and lowercase so trivial formatting differences hash identically."""
    return _WS.sub(" ", text).strip().lower()


def _duplicates(ctx: SpaceContext) -> list[Relation]:
    """Group documents by normalized-text content hash; emit ``duplicate-of`` within each group.

    Only non-empty content is considered (empty docs are not "duplicates" of each other). Within
    a group, edges are emitted in path order between consecutive documents, giving a deterministic
    chain that connects every member without an O(n^2) clique.
    """
    buckets: dict[str, list[Document]] = {}
    for doc in ctx.space.documents_:
        text = _doc_text(ctx, doc)
        if not text:
            continue
        norm = _normalize_for_hash(text)
        if not norm:
            continue
        digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        buckets.setdefault(digest, []).append(doc)

    out: list[Relation] = []
    for group in buckets.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda d: d.path)
        for a, b in zip(ordered, ordered[1:], strict=False):
            out.append(Relation(src=a.id, dst=b.id, type=RelationType.DUPLICATE_OF))
    return out

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
import re

from indx.core.context import SpaceContext
from indx.core.document import Document
from indx.core.relation import Relation, RelationType

_PART = re.compile(r"(.*?)[ _-]?(?:part[ _-]?|p)?(\d+)\b", re.IGNORECASE)
_INDEX_STEMS = {"index", "readme", "_index", "overview"}

# A markdown inline link target: the URL/path inside ``](...)``. We only keep the path part,
# dropping any title or fragment/anchor, so ``[x](docs/a.md#sec "t")`` resolves to ``docs/a.md``.
_MD_LINK = re.compile(r"\]\(\s*<?([^)>\s]+)>?(?:\s+[^)]*)?\)")

# Whitespace runs collapse to a single space when normalizing text for duplicate detection.
_WS = re.compile(r"\s+")


class RelateStage:
    name = "relate"

    def run(self, ctx: SpaceContext) -> SpaceContext:
        docs = ctx.space.documents_
        rels: list[Relation] = []
        by_folder: dict[tuple[str, ...], list[Document]] = {}
        for d in docs:
            by_folder.setdefault(tuple(d.lineage), []).append(d)

        for group in by_folder.values():
            group = sorted(group, key=lambda d: d.path)
            # sibling: every ordered pair in the same folder (high precision, symmetric pairs)
            for i, a in enumerate(group):
                for b in group[i + 1 :]:
                    rels.append(Relation(src=a.id, dst=b.id, type=RelationType.SIBLING))
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


def _stem(doc: Document) -> str:
    name = doc.path.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0]


def _continues(group: list[Document]) -> list[Relation]:
    parts: dict[str, list[tuple[int, Document]]] = {}
    for d in group:
        m = _PART.fullmatch(_stem(d))
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


def _norm_path(path: str) -> str:
    """Forward-slash, lowercased, ``./`` and trailing-slash stripped — for matching mentions."""
    p = path.replace("\\", "/").strip().lstrip("./").rstrip("/")
    return p.lower()


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
        text = _doc_text(ctx, src)
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
    if len(matches) == 1 and _norm_path(matches[0].path).endswith(cand):
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

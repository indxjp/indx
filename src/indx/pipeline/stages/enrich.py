"""05 Enrich — LLM/VLM metadata: topics, tags, summary, type (FR-S05).

The default scaffold enrichment is **LLM-free and deterministic**: it derives lightweight
topics from term frequency, a summary from the lead text, and a document **type** from cheap
signals (file extension, folder lineage keywords, markdown/heading/content cues). This keeps
the zero-config / air-gapped path fully offline (no Ollama required to get a usable space) and
keeps the corpus golden deterministic. A real LLM enricher is a swappable Stage
(testing-strategy §3.5 explains why stochastic enrichment is asserted via cassette/structure,
not byte-golden).

The configured ``[enrich] metadata`` list (technical-spec §9) selects *which* fields are
populated: the documented keys are ``type``, ``topics``, ``tags`` and ``summary`` (``type``
maps to :attr:`Document.doc_type`). A stage constructed with the default ``metadata`` populates
all four; passing a subset narrows the work without touching the others.

When an **explicit** build-time LLM is selected (see :class:`DirectoryPipeline`), the stage is
constructed with ``llm=<instance>`` and produces ``summary`` and ``topics`` from that LLM with a
deterministic (``temperature=0.0``) prompt; ``type`` and ``tags`` remain heuristic. Any per-document
LLM failure falls back to the heuristic for that document. With no explicit LLM (the zero-config /
air-gapped default) the stage is fully offline and byte-identical to the deterministic scaffold.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from indx.core.context import SpaceContext
from indx.core.parsed import ParsedDoc

# Public ``[enrich] metadata`` keys this scaffold understands. ``type`` is the config-facing
# name for the :attr:`Document.doc_type` field; the others map 1:1.
_META_TYPE = "type"
_META_TOPICS = "topics"
_META_TAGS = "tags"
_META_SUMMARY = "summary"
_DEFAULT_METADATA: tuple[str, ...] = (_META_TYPE, _META_TOPICS, _META_TAGS, _META_SUMMARY)

# Tokenizer for term-frequency topics. CJK scripts (Hiragana ぀-ヿ, Katakana, CJK ideographs
# 㐀-鿿, Hangul 가-힣) are unsegmented and never lowercased by ``str.lower``, so the original
# ASCII-only ``[a-z][a-z0-9]{3,}`` returned no tokens for Japanese/Chinese/Korean text — leaving
# ``topics`` empty for any non-Latin document. ``_WORD`` matches either a Latin word (unchanged,
# so the ASCII corpus golden is identical) **or** a *run* of consecutive CJK characters. Single
# CJK characters carry little meaning, so :func:`_tokenize` segments each CJK run into overlapping
# 2-character shingles (bigrams) — the standard dependency-free CJK segmentation heuristic — and
# falls back to the single character for a length-1 run. This keeps topics deterministic and
# stdlib-only (no jieba/mecab) while producing word-ish (not per-ideograph) Japanese/Chinese/Korean
# topics. The Latin branch is matched verbatim and emitted as-is.
_CJK_RUN = r"[぀-ヿ㐀-鿿가-힣]+"
_WORD = re.compile(r"[a-z][a-z0-9]{3,}|" + _CJK_RUN)
_CJK_CHAR = re.compile(r"[぀-ヿ㐀-鿿가-힣]")


def _tokenize(text: str) -> list[str]:
    """Term-frequency tokens for ``text`` (already lowercased by the caller).

    Latin words match the historical ``[a-z][a-z0-9]{3,}`` pattern and are emitted verbatim, so
    the ASCII corpus golden is byte-identical. A run of consecutive CJK characters is segmented
    into overlapping 2-char shingles (e.g. ``プロジェクト`` -> ``プロ``, ``ロジ``, ``ジェ`` …); a
    length-1 run emits the lone character. This is a deterministic, dependency-free approximation
    of CJK word segmentation.
    """
    tokens: list[str] = []
    for match in _WORD.findall(text):
        if _CJK_CHAR.match(match):
            if len(match) == 1:
                tokens.append(match)
            else:
                tokens.extend(match[i : i + 2] for i in range(len(match) - 1))
        else:
            tokens.append(match)
    return tokens


_STOP = {
    "this",
    "that",
    "with",
    "from",
    "have",
    "will",
    "your",
    "their",
    "about",
    "which",
    "there",
    "would",
    "these",
    "those",
    "into",
    "than",
    "then",
    "they",
    "been",
    "were",
}

# Extension → document type. Lowercased, leading dot stripped. Order is irrelevant (dict).
_EXT_TYPE: dict[str, str] = {
    "md": "markdown",
    "markdown": "markdown",
    "rst": "markdown",
    "txt": "text",
    "text": "text",
    "pdf": "pdf",
    "doc": "document",
    "docx": "document",
    "odt": "document",
    "rtf": "document",
    "ppt": "presentation",
    "pptx": "presentation",
    "key": "presentation",
    "xls": "spreadsheet",
    "xlsx": "spreadsheet",
    "csv": "spreadsheet",
    "tsv": "spreadsheet",
    "ods": "spreadsheet",
    "html": "webpage",
    "htm": "webpage",
    "json": "data",
    "yaml": "data",
    "yml": "data",
    "toml": "data",
    "xml": "data",
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "gif": "image",
    "svg": "image",
    "webp": "image",
}

# Folder-lineage keyword → document type. A folder named (or containing) one of these words
# anywhere in the lineage is a strong, deterministic signal of the document's role. Checked
# against whole, lowercased folder segments. Singular/plural variants are listed explicitly so
# matching stays an exact set membership test (no stemming, no surprises).
_LINEAGE_TYPE: dict[str, str] = {
    "policy": "policy",
    "policies": "policy",
    "contract": "contract",
    "contracts": "contract",
    "agreement": "contract",
    "agreements": "contract",
    "onboarding": "onboarding",
    "invoice": "invoice",
    "invoices": "invoice",
    "meeting": "meeting",
    "meetings": "meeting",
    "report": "report",
    "reports": "report",
    "spec": "specification",
    "specs": "specification",
}

# Content cue (substring of lowercased lead text) → document type. Lightweight last-resort
# heuristics used only when extension/lineage did not already classify a more specific role.
_CONTENT_CUES: tuple[tuple[str, str], ...] = (
    ("# readme", "readme"),
    ("readme", "readme"),
    ("meeting notes", "meeting"),
    ("meeting minutes", "meeting"),
    ("minutes of", "meeting"),
    ("agenda", "meeting"),
    ("action items", "meeting"),
    ("invoice", "invoice"),
    ("bill to", "invoice"),
    ("amount due", "invoice"),
)

# How many leading characters of the document we scan for content cues. Bounded so the stage
# stays O(1) per document regardless of file size (performance rule §14).
_CUE_SCAN = 2000

# How many leading characters of the document text are sent to the LLM enricher. Bounded so a
# huge file does not blow the prompt budget; the lead is enough for a 1-2 sentence summary.
_LLM_LEAD_CHARS = 4000

# Types whose role is inherent (extension/lineage derived) and should not be overridden by a
# weaker content cue.
_STRONG_TYPES = frozenset(_LINEAGE_TYPE.values()) | {
    "spreadsheet",
    "presentation",
    "image",
    "data",
}


class EnrichStage:
    """Deterministic, offline scaffold enrichment with type-aware metadata."""

    name = "enrich"

    def __init__(
        self,
        max_topics: int = 5,
        *,
        metadata: Sequence[str] | None = None,
        llm: object | None = None,
    ) -> None:
        """Configure the scaffold enricher.

        Args:
            max_topics: Maximum number of frequency-ranked topics per document.
            metadata: The ``[enrich] metadata`` selection — which fields to populate. Defaults
                to all scaffold-supported keys (``type``, ``topics``, ``tags``, ``summary``).
                Unknown keys are ignored so a config that names a future LLM-only field does not
                error here.
            llm: An optional LLM instance (exposing ``.complete(prompt, system=..., max_tokens=...,
                temperature=...) -> str``). When provided, ``summary`` and ``topics`` are produced
                by the LLM with a deterministic (``temperature=0.0``) prompt instead of the offline
                heuristics; ``type`` and ``tags`` stay heuristic. Any per-document LLM failure
                falls back to the heuristic for that document, so one bad doc or a creds error
                never aborts the build. When ``None`` (the zero-config / air-gapped default),
                behaviour matches the deterministic offline scaffold byte-for-byte (golden kept).
        """
        self.max_topics = max_topics
        self._llm = llm
        selected = _DEFAULT_METADATA if metadata is None else tuple(metadata)
        self._want_type = _META_TYPE in selected
        self._want_topics = _META_TOPICS in selected
        self._want_tags = _META_TAGS in selected
        self._want_summary = _META_SUMMARY in selected

    def run(self, ctx: SpaceContext) -> SpaceContext:
        """Populate selected enrichment fields on every parsed document.

        Args:
            ctx: The shared context after relate; ``ctx.parsed`` holds the parser output.

        Returns:
            The context with ``doc_type`` / ``topics`` / ``tags`` / ``summary`` set on each
            document for which a :class:`ParsedDoc` exists, per the configured metadata list.
        """
        for doc in ctx.space.documents_:
            parsed = ctx.parsed.get(doc.id)
            if parsed is None:
                continue
            text = parsed.text
            doc_type = self._detect_type(doc.path, doc.lineage, parsed)
            if self._want_type:
                doc.doc_type = doc_type
            # When an explicit LLM is wired in, ask it once per document for the summary + topics
            # (a single deterministic prompt yields both). ``None`` means stay offline/heuristic.
            llm_summary: str | None = None
            llm_topics: list[str] | None = None
            if self._llm is not None and (self._want_summary or self._want_topics):
                llm_summary, llm_topics = self._llm_enrich(parsed, doc_type)
            if self._want_topics:
                doc.topics = llm_topics if llm_topics is not None else self._topics(text)
            if self._want_summary:
                doc.summary = (
                    llm_summary if llm_summary is not None else self._summary(parsed, doc_type)
                )
            if self._want_tags:
                # Tags are type-aware: they always carry the detected role, then folder-lineage
                # cues, so downstream filters can pivot on either without re-deriving them.
                doc.tags = self._tags(doc_type, doc.lineage)
        return ctx

    def _detect_type(self, path: str, lineage: Sequence[str], parsed: ParsedDoc) -> str:
        """Classify a document from extension, lineage and content cues (deterministic).

        Precedence: a role named in the folder lineage (e.g. ``contracts/``) wins, because the
        author filed it there deliberately; otherwise a structural extension type (spreadsheet,
        image, …) or markdown heading structure; finally a content cue; falling back to the bare
        extension type or ``"unknown"``.
        """
        ext_type = _ext_type(path)

        # 1. Lineage role is the most intentional signal.
        for segment in lineage:
            mapped = _LINEAGE_TYPE.get(segment.strip().lower())
            if mapped is not None:
                return mapped

        # 2. A strong structural extension type (spreadsheet/image/presentation/data) is more
        #    reliable than scanning text for cues.
        if ext_type in _STRONG_TYPES:
            return ext_type

        # 3. Markdown with heading structure (from extension or detected heading blocks).
        has_heading = any(b.kind == "heading" for b in parsed.blocks)
        lead = _lead(parsed.text)
        if ext_type == "markdown" or has_heading or lead.lstrip().startswith("#"):
            cue = _content_cue(lead)
            # A README/meeting markdown doc keeps its more specific role.
            return cue if cue is not None else "markdown"

        # 4. Lightweight content cues for plain/unknown types.
        cue = _content_cue(lead)
        if cue is not None:
            return cue

        # 5. Fall back to whatever the extension implies, else unknown.
        return ext_type or "unknown"

    def _topics(self, text: str) -> list[str]:
        counts = Counter(w for w in _tokenize(text.lower()) if w not in _STOP)
        # deterministic: sort by (-count, word)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [w for w, _ in ranked[: self.max_topics]]

    def _tags(self, doc_type: str, lineage: Sequence[str]) -> list[str]:
        """Type-aware tags: the detected role plus distinct lineage segments, order-stable.

        Lineage segments that are themselves role keywords (e.g. ``policies`` → ``policy``) are
        normalized to the canonical role so they collapse against the detected ``doc_type``
        instead of producing a near-duplicate tag.
        """
        tags: list[str] = []
        seen: set[str] = set()
        candidates = [doc_type]
        for seg in lineage:
            norm = seg.strip().lower()
            candidates.append(_LINEAGE_TYPE.get(norm, norm))
        for candidate in candidates:
            if candidate and candidate != "unknown" and candidate not in seen:
                seen.add(candidate)
                tags.append(candidate)
        return tags

    def _summary(self, parsed: ParsedDoc, doc_type: str, limit: int = 200) -> str:
        """Lead-text summary, made type-aware by preferring the first heading as a title.

        For heading-structured documents (markdown, reports) the summary leads with the first
        heading so the one-line preview reads as a title + body rather than a truncated header.
        """
        heading = _first_heading(parsed)
        if heading:
            heading = " ".join(heading.split()).lstrip("# ").strip()
            # ``parsed.text`` joins every block, including the heading block whose text
            # still carries the leading '#' markers. Build the body from the non-heading
            # blocks so the heading is not repeated and no '#' leaks into the summary.
            body = " ".join(
                " ".join(b.text.split())
                for b in sorted(parsed.blocks, key=lambda b: b.order)
                if b.kind != "heading"
            ).strip()
            if heading and not body.startswith(heading):
                combined = f"{heading} — {body}" if body else heading
                return combined[:limit]
            return body[:limit]
        return " ".join(parsed.text.split())[:limit]

    def _llm_enrich(self, parsed: ParsedDoc, doc_type: str) -> tuple[str | None, list[str] | None]:
        """Ask the wired LLM for a summary + topic list from the document's lead text.

        Returns ``(summary, topics)`` where each element is ``None`` if not wanted or if the LLM
        call/parse failed (the caller then falls back to the heuristic for that field). Both the
        call and the parse are wrapped so any failure — a creds/network error, a malformed reply,
        an unexpected shape — degrades to the offline heuristic instead of aborting the build.
        """
        llm = self._llm
        if llm is None:
            return None, None
        lead = " ".join(parsed.text.split())[:_LLM_LEAD_CHARS]
        if not lead:
            return None, None
        system = (
            "You summarize documents for a search index. Reply with EXACTLY two lines and nothing "
            "else.\nLine 1: SUMMARY: <a 1-2 sentence neutral summary>\n"
            "Line 2: TOPICS: <a short comma-separated list of key topics>"
        )
        prompt = f"Document type: {doc_type}\n\nDocument text:\n{lead}"
        try:
            complete = llm.complete  # type: ignore[attr-defined]
            reply = str(complete(prompt, system=system, max_tokens=256, temperature=0.0))
        except Exception:  # creds/network/adapter failure → heuristic fallback (do not crash)
            return None, None
        return self._parse_llm_reply(reply)

    def _parse_llm_reply(self, reply: str) -> tuple[str | None, list[str] | None]:
        """Parse the ``SUMMARY:`` / ``TOPICS:`` lines from the LLM reply (resilient to noise)."""
        summary: str | None = None
        topics: list[str] | None = None
        for raw in reply.splitlines():
            line = raw.strip()
            lower = line.lower()
            if lower.startswith("summary:"):
                value = line.split(":", 1)[1].strip()
                if value:
                    summary = value
            elif lower.startswith("topics:"):
                value = line.split(":", 1)[1].strip()
                parsed_topics = [t.strip() for t in value.split(",") if t.strip()]
                if parsed_topics:
                    topics = parsed_topics[: self.max_topics]
        if self._want_summary and summary is None and self._want_topics and topics is None:
            # Nothing usable parsed out — let both fields fall back to the heuristic.
            return None, None
        return (
            (summary if self._want_summary else None),
            (topics if self._want_topics else None),
        )


def _ext_type(path: str) -> str:
    """Return the document type implied by ``path``'s extension, or ``""`` if unknown."""
    _, _, ext = path.rpartition(".")
    if not ext or ext == path:
        return ""
    return _EXT_TYPE.get(ext.strip().lower(), "")


def _lead(text: str) -> str:
    """First :data:`_CUE_SCAN` characters, lowercased — the window scanned for content cues."""
    return text[:_CUE_SCAN].lower()


def _content_cue(lead: str) -> str | None:
    """Return the first matching content-cue type for the lowercased lead text, else ``None``."""
    for needle, mapped in _CONTENT_CUES:
        if needle in lead:
            return mapped
    return None


def _first_heading(parsed: ParsedDoc) -> str | None:
    """The text of the earliest heading block in document order, if any."""
    headings = [b for b in parsed.blocks if b.kind == "heading"]
    if not headings:
        return None
    return min(headings, key=lambda b: b.order).text

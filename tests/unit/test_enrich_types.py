"""T1 — type-aware deterministic Enrich (FR-S05, type/topics/tags/summary).

Exercises the LLM-free scaffold enrichment: deterministic document-type detection from file
extension, folder lineage and content cues, plus the type-aware topics/tags/summary behaviour
and the ``[enrich] metadata`` selection. No network/LLM is involved.
"""

from __future__ import annotations

from collections.abc import Sequence

from indx.core.context import SpaceContext
from indx.core.document import Document
from indx.core.knowledge_space import KnowledgeSpace
from indx.core.parsed import Block, ParsedDoc
from indx.pipeline.stages.enrich import EnrichStage


def _ctx(doc: Document, parsed: ParsedDoc) -> SpaceContext:
    space = KnowledgeSpace()
    space.documents_.append(doc)
    ctx = SpaceContext(root="/tmp", space=space)
    ctx.parsed[doc.id] = parsed
    return ctx


def _parsed(*blocks: Block, source_path: str = "x") -> ParsedDoc:
    return ParsedDoc(source_path=source_path, blocks=list(blocks))


def _enrich(
    doc: Document,
    parsed: ParsedDoc,
    *,
    metadata: Sequence[str] | None = None,
) -> Document:
    ctx = _ctx(doc, parsed)
    EnrichStage(metadata=metadata).run(ctx)
    return ctx.space.documents_[0]


# --- type detection ---------------------------------------------------------------------


def test_lineage_role_wins_over_structural_extension() -> None:
    doc = Document(id="d1", path="reports/q1.xlsx", lineage=["reports"])
    out = _enrich(doc, _parsed(Block(text="numbers", order=0)))
    # A spreadsheet extension is a strong structural type, but a deliberate lineage role
    # ("reports/") wins by design. This asserts the contract precedence.
    assert out.doc_type == "report"


def test_spreadsheet_extension_without_lineage_role() -> None:
    doc = Document(id="d1", path="data/export.csv")
    out = _enrich(doc, _parsed(Block(text="a,b,c", order=0)))
    assert out.doc_type == "spreadsheet"


def test_markdown_extension_detected() -> None:
    doc = Document(id="d1", path="guide.md")
    out = _enrich(doc, _parsed(Block(kind="text", text="plain body", order=0)))
    assert out.doc_type == "markdown"


def test_lineage_keyword_classifies_contract() -> None:
    doc = Document(id="d1", path="legal/msa.pdf", lineage=["legal", "contracts"])
    out = _enrich(doc, _parsed(Block(text="terms and conditions", order=0)))
    assert out.doc_type == "contract"


def test_lineage_keyword_classifies_policy_and_onboarding() -> None:
    policy = _enrich(
        Document(id="p", path="hr/leave.txt", lineage=["policies"]),
        _parsed(Block(text="leave policy", order=0)),
    )
    onboarding = _enrich(
        Document(id="o", path="hr/day1.txt", lineage=["onboarding"]),
        _parsed(Block(text="welcome aboard", order=0)),
    )
    assert policy.doc_type == "policy"
    assert onboarding.doc_type == "onboarding"


def test_heading_block_implies_markdown() -> None:
    doc = Document(id="d1", path="notes")  # no extension
    parsed = _parsed(
        Block(kind="heading", text="# Overview", order=0),
        Block(kind="text", text="body", order=1),
    )
    assert _enrich(doc, parsed).doc_type == "markdown"


def test_content_cue_readme() -> None:
    doc = Document(id="d1", path="README")
    parsed = _parsed(Block(kind="heading", text="# README", order=0))
    assert _enrich(doc, parsed).doc_type == "readme"


def test_content_cue_meeting_notes() -> None:
    doc = Document(id="d1", path="2026-06-01.txt")
    parsed = _parsed(Block(text="Meeting notes from the sync", order=0))
    assert _enrich(doc, parsed).doc_type == "meeting"


def test_content_cue_invoice() -> None:
    doc = Document(id="d1", path="acme.txt")
    parsed = _parsed(Block(text="Invoice\nBill to: Acme\nAmount due: 100", order=0))
    assert _enrich(doc, parsed).doc_type == "invoice"


def test_unknown_type_falls_back() -> None:
    doc = Document(id="d1", path="mystery.bin")
    out = _enrich(doc, _parsed(Block(text="opaque payload", order=0)))
    assert out.doc_type == "unknown"


def test_detection_is_deterministic() -> None:
    doc = Document(id="d1", path="legal/msa.pdf", lineage=["contracts"])
    parsed = _parsed(Block(text="terms", order=0))
    first = _enrich(doc.model_copy(deep=True), parsed).doc_type
    second = _enrich(doc.model_copy(deep=True), parsed).doc_type
    assert first == second == "contract"


# --- type-aware tags --------------------------------------------------------------------


def test_tags_carry_type_then_lineage() -> None:
    doc = Document(id="d1", path="hr/leave.txt", lineage=["hr", "policies"])
    out = _enrich(doc, _parsed(Block(text="leave policy", order=0)))
    assert out.tags == ["policy", "hr"]  # type first, lineage next, dedup'd against type


def test_tags_dedup_against_type() -> None:
    doc = Document(id="d1", path="x.txt", lineage=["onboarding"])
    out = _enrich(doc, _parsed(Block(text="welcome", order=0)))
    # doc_type == "onboarding" and lineage has "onboarding"; it must not appear twice.
    assert out.tags == ["onboarding"]


def test_tags_skip_unknown_type() -> None:
    doc = Document(id="d1", path="mystery.bin")
    out = _enrich(doc, _parsed(Block(text="opaque", order=0)))
    assert out.tags == []  # "unknown" is not emitted as a tag


# --- type-aware summary -----------------------------------------------------------------


def test_summary_leads_with_first_heading() -> None:
    doc = Document(id="d1", path="guide.md")
    parsed = _parsed(
        Block(kind="heading", text="# Onboarding Guide", order=0),
        Block(kind="text", text="Welcome to the team.", order=1),
    )
    out = _enrich(doc, parsed)
    assert out.summary is not None
    assert out.summary.startswith("Onboarding Guide — ")
    assert "Welcome to the team." in out.summary


def test_summary_without_heading_is_lead_text() -> None:
    doc = Document(id="d1", path="notes.txt")
    parsed = _parsed(Block(text="Just a plain note with no headings.", order=0))
    out = _enrich(doc, parsed)
    assert out.summary == "Just a plain note with no headings."


# --- metadata selection -----------------------------------------------------------------


def test_metadata_subset_only_populates_selected_fields() -> None:
    doc = Document(id="d1", path="guide.md", lineage=["onboarding"])
    parsed = _parsed(
        Block(kind="heading", text="# Title", order=0),
        Block(kind="text", text="body words words", order=1),
    )
    out = _enrich(doc, parsed, metadata=["type"])
    assert out.doc_type == "onboarding"
    assert out.topics == []  # not selected
    assert out.tags == []  # not selected
    assert out.summary is None  # not selected


def test_unknown_metadata_key_is_ignored() -> None:
    doc = Document(id="d1", path="guide.md")
    parsed = _parsed(Block(kind="text", text="alpha beta gamma", order=0))
    # "references" is a future LLM-only field; the scaffold must not error on it.
    out = _enrich(doc, parsed, metadata=["references", "topics"])
    assert out.topics  # topics still populated
    assert out.doc_type is None  # type not selected


def test_default_metadata_populates_all_four() -> None:
    doc = Document(id="d1", path="guide.md", lineage=["onboarding"])
    parsed = _parsed(
        Block(kind="heading", text="# Welcome", order=0),
        Block(kind="text", text="onboarding handbook content here", order=1),
    )
    out = _enrich(doc, parsed)
    assert out.doc_type == "onboarding"
    assert out.topics
    assert out.tags == ["onboarding"]
    assert out.summary and out.summary.startswith("Welcome — ")

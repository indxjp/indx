"""Feature 4 — ``KnowledgeSpace.ask`` (offline extractive + configured-LLM synthesis)."""

from __future__ import annotations

from pathlib import Path

from indx import Answer, DirectoryPipeline, KnowledgeSpace


def _offline_space(tmp_path: Path) -> KnowledgeSpace:
    src = tmp_path / "src"
    src.mkdir()
    (src / "policy.txt").write_text(
        "# Leave Policy\n\nEmployees may take paid leave. Requests go to the manager.\n",
        encoding="utf-8",
    )
    (src / "guide.txt").write_text(
        "# Onboarding Guide\n\nWelcome to the team and the company.\n", encoding="utf-8"
    )
    pipe = DirectoryPipeline(
        parser="plaintext", llm="none", embedder="hash", store="jsonl", output="jsonl"
    )
    return pipe.run(str(src), str(tmp_path / "out.jsonl"))


def test_ask_offline_extractive_with_citations(tmp_path: Path) -> None:
    space = _offline_space(tmp_path)
    answer = space.ask("leave policy")
    assert isinstance(answer, Answer)
    assert answer.llm == "none"
    assert answer.hits
    assert "[1]" in answer.answer
    # M1: the offline answer text must NOT embed its own "Sources:" block — the CLI renders the
    # source list once from Answer.sources. Embedding it here caused citations to print twice.
    assert "Sources:" not in answer.answer
    assert any("policy.txt" in s for s in answer.sources)


def test_ask_offline_extracts_relevant_sentence_not_raw_chunk(tmp_path: Path) -> None:
    # M2: the body surfaces a query-relevant extracted sentence, not the whole raw chunk dump.
    space = _offline_space(tmp_path)
    answer = space.ask("paid leave manager")
    assert "leave" in answer.answer.lower()


def test_ask_offline_neutralizes_table_pipes(tmp_path: Path) -> None:
    # M2: pipe-table rows are neutralized so Rich never parses the body as a (blank) table.
    src = tmp_path / "tbl"
    src.mkdir()
    (src / "rows.txt").write_text(
        "| name | role |\n| alice | manager |\n| bob | analyst |\n",
        encoding="utf-8",
    )
    pipe = DirectoryPipeline(
        parser="plaintext", llm="none", embedder="hash", store="jsonl", output="jsonl"
    )
    space = pipe.run(str(src), str(tmp_path / "out.jsonl"))
    answer = space.ask("manager")
    assert "| name | role |" not in answer.answer
    assert "|" not in answer.answer


def test_ask_empty_space() -> None:
    answer = KnowledgeSpace().ask("anything")
    assert answer.answer == "No matching content found in the space."
    assert answer.hits == []
    assert answer.llm == "none"


def test_ask_deterministic(tmp_path: Path) -> None:
    space = _offline_space(tmp_path)
    a = space.ask("onboarding")
    b = space.ask("onboarding")
    assert a.answer == b.answer
    assert a.sources == b.sources


def test_ask_respects_k(tmp_path: Path) -> None:
    space = _offline_space(tmp_path)
    answer = space.ask("leave", k=1)
    assert len(answer.hits) <= 1


def test_ask_k_zero_is_safe(tmp_path: Path) -> None:
    space = _offline_space(tmp_path)
    answer = space.ask("leave", k=0)
    assert answer.answer == ""
    assert answer.hits == []


def test_ask_uses_configured_llm(tmp_path: Path, monkeypatch) -> None:
    """When the manifest selects a real llm, ``ask`` routes through ``get_llm`` and records it."""
    space = _offline_space(tmp_path)
    space.manifest.components["llm"] = "fakellm"

    class _FakeLLM:
        name = "fakellm"

        def complete(self, prompt: str, **_kwargs: object) -> str:
            return "SYNTHESIZED-ANSWER"

    import indx.core.knowledge_space as ks_mod

    monkeypatch.setattr("indx.registry.get_llm", lambda name, **_kw: _FakeLLM(), raising=True)
    # The import inside ask resolves indx.registry.get_llm at call time.
    assert ks_mod  # silence unused-import linters; module imported for clarity
    answer = space.ask("leave policy")
    assert answer.llm == "fakellm"
    assert answer.answer == "SYNTHESIZED-ANSWER"
    assert answer.hits  # retrieval still happened

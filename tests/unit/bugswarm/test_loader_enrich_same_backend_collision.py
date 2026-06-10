"""Config loader — same-backend ``enrich`` env passthrough must not silently mis-route.

The ``enrich`` section hosts two adapter slots (``llm`` + ``vlm``). Their ``INDX_LLM__*`` and
``INDX_VLM__*`` passthrough is namespaced under a per-backend ``[enrich.<backend>]`` sub-table.
That keeps the slots apart only while they resolve to *different* backends. When both resolve to
the **same** backend (e.g. both Azure OpenAI) the single shared sub-table would silently clobber
one slot's secret/endpoint and cross-leak it to the other, since ``Config.slot_options()`` keys
purely on the backend name. The loader must reject this instead of misrouting credentials.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indx.config.loader import load_config
from indx.errors import ConfigError


def test_same_backend_distinct_passthrough_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """llm + vlm on one backend with separate INDX_*__* passthrough is a hard error.

    Without the guard this silently produced ``slot_options()['llm'] == slot_options()['vlm']``
    (the vlm secret overwriting and replacing the lost llm secret).
    """
    monkeypatch.setenv("INDX_LLM__API_KEY", "llm-secret")
    monkeypatch.setenv("INDX_LLM__BASE_URL", "https://llm.example")
    monkeypatch.setenv("INDX_VLM__API_KEY", "vlm-secret")
    monkeypatch.setenv("INDX_VLM__BASE_URL", "https://vlm.example")

    with pytest.raises(ConfigError) as excinfo:
        load_config(overrides={"llm": "azure:gpt-4", "vlm": "azure:gpt-4o"})

    message = str(excinfo.value)
    assert "azure" in message
    assert "llm" in message and "vlm" in message
    # The leaked secret must never appear in the actionable error.
    assert "secret" not in message


def test_same_backend_via_toml_selector_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Backend resolution from the TOML selector also triggers the guard."""
    (tmp_path / "indx.toml").write_text(
        '[enrich]\nllm = "azure:gpt-4"\nvlm = "azure:gpt-4o"\n', encoding="utf-8"
    )
    monkeypatch.setenv("INDX_LLM__API_KEY", "llm-secret")
    monkeypatch.setenv("INDX_VLM__API_KEY", "vlm-secret")

    with pytest.raises(ConfigError, match="enrich"):
        load_config(cwd=tmp_path)


def test_distinct_backends_still_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """The non-colliding case (distinct backends) keeps working — no false positive."""
    monkeypatch.setenv("INDX_LLM__API_KEY", "llm-secret")
    monkeypatch.setenv("INDX_VLM__API_KEY", "vlm-secret")

    opts = load_config(overrides={"llm": "openai:gpt-4", "vlm": "ollama:llava"}).slot_options()

    assert opts["llm"] == {"api_key": "llm-secret"}
    assert opts["vlm"] == {"api_key": "vlm-secret"}


def test_same_backend_single_slot_passthrough_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only one slot supplying passthrough on a shared backend is unambiguous and allowed."""
    monkeypatch.setenv("INDX_LLM__API_KEY", "llm-secret")

    opts = load_config(overrides={"llm": "azure:gpt-4", "vlm": "azure:gpt-4o"}).slot_options()

    assert opts["llm"] == {"api_key": "llm-secret"}

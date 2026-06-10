"""Config loader — env layering for the dual-slot ``enrich`` section.

The ``enrich`` section hosts two adapter slots (``llm`` + ``vlm``). Their ``INDX_LLM__*``
and ``INDX_VLM__*`` passthrough env vars must each reach their own adapter without
colliding or cross-leaking, mirroring the namespaced ``[enrich.<backend>]`` TOML shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indx.config.loader import load_config


def test_env_llm_vlm_passthrough_no_collision_no_crossleak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INDX_LLM__* and INDX_VLM__* with shared keys each scope to their own adapter.

    With the default selectors (llm -> ``openai``, vlm -> ``none``) the two slots resolve
    to distinct backends, so their same-named keys (``base_url``, ``api_key``) must not
    overwrite one another, and an llm-only key (``model``) must not leak into the vlm
    adapter.
    """
    monkeypatch.setenv("INDX_LLM__BASE_URL", "http://llm-host/v1")
    monkeypatch.setenv("INDX_VLM__BASE_URL", "http://vlm-host/v1")
    monkeypatch.setenv("INDX_LLM__MODEL", "llm-model")
    monkeypatch.setenv("INDX_LLM__API_KEY", "llm-key")
    monkeypatch.setenv("INDX_VLM__API_KEY", "vlm-key")

    opts = load_config().slot_options()

    assert opts["llm"] == {
        "base_url": "http://llm-host/v1",
        "model": "llm-model",
        "api_key": "llm-key",
    }
    assert opts["vlm"] == {
        "base_url": "http://vlm-host/v1",
        "api_key": "vlm-key",
    }


def test_env_llm_passthrough_namespaces_under_toml_selector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Env passthrough lands under the backend resolved from the TOML selector."""
    (tmp_path / "indx.toml").write_text('[enrich]\nllm = "anthropic:claude"\n', encoding="utf-8")
    monkeypatch.setenv("INDX_LLM__API_KEY", "env-key")

    cfg = load_config(cwd=tmp_path)

    assert cfg.enrich.llm == "anthropic:claude"
    assert cfg.slot_options()["llm"] == {"api_key": "env-key"}


def test_cli_override_backend_keeps_env_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CLI ``--llm`` override that changes the backend must keep INDX_LLM__* passthrough.

    The override is the highest-precedence layer, so it picks the effective backend. The
    env passthrough must be namespaced under *that* backend, not the env/toml/default one,
    otherwise ``slot_options()["llm"]`` (which reads the sub-table for the resolved backend)
    silently drops the env-supplied secret.
    """
    monkeypatch.setenv("INDX_LLM__API_KEY", "env-key")

    cfg = load_config(overrides={"llm": "anthropic:claude"})

    assert cfg.enrich.llm == "anthropic:claude"
    assert cfg.slot_options()["llm"] == {"api_key": "env-key"}


def test_cloud_preset_override_keeps_vlm_env_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A preset-style ``--vlm`` override (folded into overrides) keeps INDX_VLM__* options."""
    monkeypatch.setenv("INDX_VLM__API_KEY", "vlm-env-key")
    monkeypatch.setenv("INDX_VLM__BASE_URL", "http://vlm/v1")

    cfg = load_config(overrides={"vlm": "azure:gpt-4o"})

    assert cfg.enrich.vlm == "azure:gpt-4o"
    assert cfg.slot_options()["vlm"] == {
        "api_key": "vlm-env-key",
        "base_url": "http://vlm/v1",
    }


def test_toml_path_namespaces_llm_and_vlm_by_subtable(tmp_path: Path) -> None:
    """The TOML ``[enrich.<backend>]`` path still resolves per-adapter options."""
    (tmp_path / "indx.toml").write_text(
        """
[enrich]
llm = "openai:gpt-4"
vlm = "ollama:llava"

[enrich.openai]
api_key = "openai-key"
base_url = "http://openai/v1"

[enrich.ollama]
base_url = "http://ollama/v1"
""",
        encoding="utf-8",
    )

    opts = load_config(cwd=tmp_path).slot_options()

    assert opts["llm"] == {"api_key": "openai-key", "base_url": "http://openai/v1"}
    assert opts["vlm"] == {"base_url": "http://ollama/v1"}

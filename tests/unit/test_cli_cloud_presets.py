"""CLI surface for the --aws / --azure / --gcp single-cloud stack presets.

Each preset is a sibling of --offline: it fills only the slots the user did NOT pass
explicitly, BEFORE load_config, so explicit flags still win and schema defaults are never
mutated (cloud-providers-design §7). The presets are mutually exclusive with each other and
with --offline. These tests use the CliRunner + --dry-run pattern (no network/model touched):
--dry-run walks and prints the resolved component plan, which is exactly what we assert on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import indx.cli.build as build_mod
from indx.cli.app import app
from indx.cli.build import _CLOUD_STACKS

runner = CliRunner()


def _src(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("# A\n\nbody\n")
    return src


def _captured_overrides(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> dict[str, Any]:
    """Run ``build`` and capture the slot overrides handed to ``load_config``.

    The preset fill happens BEFORE ``load_config`` (so values flow through the normal override
    precedence), so the captured dict is exactly the resolved slot selection — without needing
    the cloud SDKs installed to construct the real adapters. ``load_config`` is stubbed to raise
    after capture so no pipeline construction (and thus no MissingExtraError) is reached.
    """
    captured: dict[str, Any] = {}

    def fake_load_config(config: Any, *, overrides: dict[str, Any]) -> Any:
        captured.update(overrides)
        raise SystemExit(0)  # stop cleanly before DirectoryPipeline construction

    monkeypatch.setattr(build_mod, "load_config", fake_load_config)
    runner.invoke(app, argv)
    return captured


@pytest.mark.parametrize("cloud", ["aws", "azure", "gcp"])
def test_preset_fills_unset_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cloud: str
) -> None:
    """Each preset selects its whole stack for the unset slots (parser/llm/embedder/store)."""
    src = _src(tmp_path)
    overrides = _captured_overrides(
        monkeypatch, ["build", str(src), "--out", str(tmp_path / "o"), f"--{cloud}"]
    )
    stack = _CLOUD_STACKS[cloud]
    for slot in ("parser", "llm", "embedder", "store"):
        assert overrides[slot] == stack[slot]
    # VLM stays unset by every preset (opt in with --vlm).
    assert overrides["vlm"] is None


def test_explicit_flag_wins_over_preset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit --parser overrides the preset's choice for that slot only."""
    src = _src(tmp_path)
    overrides = _captured_overrides(
        monkeypatch,
        ["build", str(src), "--out", str(tmp_path / "o"), "--aws", "--parser", "plaintext"],
    )
    assert overrides["parser"] == "plaintext"  # explicit flag wins
    assert overrides["llm"] == "bedrock"  # other slots still filled by the preset
    assert overrides["embedder"] == "bedrock"
    assert overrides["store"] == "s3vectors"


def test_two_presets_is_usage_error(tmp_path: Path) -> None:
    """Passing two cloud presets is a Typer usage error (exit code 2)."""
    src = _src(tmp_path)
    result = runner.invoke(
        app, ["build", str(src), "--out", str(tmp_path / "o"), "--aws", "--gcp", "--dry-run"]
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_preset_plus_offline_is_usage_error(tmp_path: Path) -> None:
    """Mixing a cloud preset with --offline is a Typer usage error (exit code 2)."""
    src = _src(tmp_path)
    result = runner.invoke(
        app, ["build", str(src), "--out", str(tmp_path / "o"), "--aws", "--offline", "--dry-run"]
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_build_help_lists_cloud_flags() -> None:
    result = runner.invoke(app, ["build", "--help"])
    assert result.exit_code == 0
    for flag in ("--aws", "--azure", "--gcp"):
        assert flag in result.stdout

"""Schema-level tests for ``_SlotConfig.options`` passthrough semantics.

Regression coverage for the bug where a dict-valued *top-level* adapter option (e.g.
``headers`` / ``client_kwargs``) was indistinguishable from a backend sub-table and got
silently dropped, never reaching the adapter constructor.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from indx.config.schema import Config, StoreConfig, WalkConfig


def test_options_forwards_dict_valued_passthrough_option() -> None:
    """A dict option whose key is *not* a backend name must reach the adapter."""
    cfg = StoreConfig.model_validate(
        {
            "backend": "qdrant",
            "url": "http://localhost:6333",
            # dict-valued passthrough option (not a backend sub-table)
            "headers": {"Authorization": "Bearer tok"},
            # the active backend's sub-table; merged in
            "qdrant": {"collection": "docs"},
        }
    )

    opts = cfg.options()

    assert opts["url"] == "http://localhost:6333"
    assert opts["headers"] == {"Authorization": "Bearer tok"}
    assert opts["collection"] == "docs"


def test_options_drops_inactive_backend_subtable_but_keeps_dict_option() -> None:
    """Sibling backend sub-tables are dropped; dict options for other keys are kept."""
    cfg = StoreConfig.model_validate(
        {
            "backend": "qdrant",
            "qdrant": {"collection": "docs"},
            # sub-table for a backend that is *not* active -> dropped
            "pgvector": {"dsn": "postgres://x"},
            # dict-valued passthrough option -> forwarded
            "metadata_filters": {"type": "note"},
        }
    )

    opts = cfg.options()

    assert opts["collection"] == "docs"
    assert opts["metadata_filters"] == {"type": "note"}
    assert "dsn" not in opts
    assert "pgvector" not in opts


def test_slot_options_preserves_store_dict_option() -> None:
    """End-to-end via ``Config.slot_options`` the dict option survives."""
    cfg = Config.model_validate(
        {
            "store": {
                "backend": "qdrant",
                "client_kwargs": {"timeout": 30},
            }
        }
    )

    assert cfg.slot_options()["store"]["client_kwargs"] == {"timeout": 30}


# ── [walk] section (Feature 5) ───────────────────────────────────────────────


def test_walk_section_validates_and_to_filter_parses_size() -> None:
    """``[walk] max_size = "2mb"`` validates and ``to_filter()`` parses the suffix to bytes."""
    cfg = Config.model_validate({"walk": {"ext": ["md"], "max_size": "2mb"}})
    assert isinstance(cfg.walk, WalkConfig)
    flt = cfg.walk.to_filter()
    assert flt.ext == [".md"]
    assert flt.max_size == 2097152


def test_walk_default_is_empty_filter() -> None:
    """A Config with no [walk] section yields an empty (no-op) filter."""
    cfg = Config.model_validate({})
    assert cfg.walk.to_filter().is_empty()


def test_unknown_walk_key_is_rejected() -> None:
    """``extra="forbid"`` turns a typo'd [walk] key into a validation error."""
    with pytest.raises(ValidationError):
        Config.model_validate({"walk": {"exclud": ["x"]}})

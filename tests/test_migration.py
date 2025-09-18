from __future__ import annotations

import json
from pathlib import Path

import pytest

from entropy_news.utils.migration import legacy_args_to_config, migrate_legacy_json


def test_legacy_args_to_config_defaults() -> None:
    config = legacy_args_to_config({"embed_dim": 200, "hidden_dim": 32})
    assert config.architecture == "lstm"
    assert config.embed_dim == 200
    assert config.hidden_dim == 32
    assert config.dropout == pytest.approx(0.1)


def test_migrate_legacy_json(tmp_path: Path) -> None:
    payload = {"embed_dim": 128, "hidden_dim": 64, "architecture": "transformer"}
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps(payload))
    destination = tmp_path / "config.json"

    config = migrate_legacy_json(source, destination)
    assert config.architecture == "transformer"
    assert destination.exists()
    loaded = json.loads(destination.read_text())
    assert loaded["embed_dim"] == 128

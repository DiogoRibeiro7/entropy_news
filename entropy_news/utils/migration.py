"""Utilities for migrating legacy checkpoints and CLI arguments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from entropy_news.model import ModelConfig

LEGACY_DEFAULTS = {
    "architecture": "lstm",
    "vocab_size": 10000,
    "embed_dim": 100,
    "hidden_dim": 16,
    "num_heads": 2,
    "ff_dim": 128,
    "num_layers": 2,
    "dropout": 0.1,
}


def legacy_args_to_config(
    legacy_args: dict[str, Any],
    *,
    overrides: dict[str, Any] | None = None,
) -> ModelConfig:
    """Convert legacy CLI arguments into a :class:`ModelConfig` instance.

    Args:
        legacy_args: Mapping produced by older training scripts. Expected keys
            include ``embed_dim``, ``hidden_dim``, ``num_layers`` and ``dropout``.
        overrides: Optional explicit overrides applied on top of the converted
            configuration. This enables callers to patch missing information such
            as ``architecture`` or updated vocabulary sizes.

    Returns:
        A validated :class:`ModelConfig` ready for use with :class:`ModelFactory`.
    """

    params = {**LEGACY_DEFAULTS, **legacy_args}
    if overrides:
        params.update(overrides)
    config = ModelConfig(
        architecture=str(params["architecture"]),
        vocab_size=int(params["vocab_size"]),
        embed_dim=int(params["embed_dim"]),
        hidden_dim=int(params["hidden_dim"]),
        num_heads=int(params["num_heads"]),
        ff_dim=int(params["ff_dim"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params["dropout"]),
    )
    config.validate()
    return config


def migrate_legacy_json(
    source: str | Path,
    destination: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> ModelConfig:
    """Read a legacy JSON blob and emit a :class:`ModelConfig`.

    Args:
        source: Path to the legacy JSON file containing hyperparameters.
        destination: Optional path where the resulting configuration is stored.
        overrides: Optional dictionary with override values applied after the
            migration. This mirrors :func:`legacy_args_to_config`.

    Returns:
        The migrated :class:`ModelConfig` instance.
    """

    data = json.loads(Path(source).read_text())
    config = legacy_args_to_config(data, overrides=overrides)
    if destination is not None:
        config.save(destination)
    return config

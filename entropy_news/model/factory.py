"""Factory for constructing model instances from configuration objects."""

from __future__ import annotations

from typing import Protocol

from .config import ModelConfig

try:  # Optional torch dependency
    from .lstm_entropy import EntropyLSTM
    from .transformer_entropy import EntropyTransformer
except Exception:  # pragma: no cover - when torch missing
    EntropyLSTM = None  # type: ignore
    EntropyTransformer = None  # type: ignore


class SupportsForward(Protocol):
    """Simplified protocol for PyTorch-like models."""

    def forward(self, *args, **kwargs):  # pragma: no cover - protocol def
        ...


class ModelFactory:
    """Create model instances based on :class:`ModelConfig` values."""

    @staticmethod
    def create(config: ModelConfig) -> SupportsForward:
        """Instantiate a model described by ``config``.

        Args:
            config: Configuration describing the desired model.

        Returns:
            A model instance matching ``config``.
        """
        config.validate()
        if config.architecture == "lstm":
            if EntropyLSTM is None:
                raise RuntimeError("EntropyLSTM requires torch to be installed")
            return EntropyLSTM(
                vocab_size=config.vocab_size,
                embed_dim=config.embed_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
            )
        if config.architecture == "transformer":
            if EntropyTransformer is None:
                raise RuntimeError("EntropyTransformer requires torch to be installed")
            return EntropyTransformer(
                vocab_size=config.vocab_size,
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                ff_dim=config.ff_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
            )
        raise ValueError(f"Unknown architecture: {config.architecture}")

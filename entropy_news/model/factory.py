"""Factory for constructing model instances from configuration objects."""

from __future__ import annotations

from typing import Any, Protocol

from .config import ModelConfig

try:  # Optional torch dependency
    from .lstm_entropy import EntropyLSTM
    from .lstm_attention import EntropyLSTMAttention
    from .transformer_entropy import EntropyTransformer
except Exception:  # pragma: no cover - when torch missing
    EntropyLSTM = None  # type: ignore
    EntropyLSTMAttention = None  # type: ignore
    EntropyTransformer = None  # type: ignore


class SupportsForward(Protocol):
    """Simplified protocol for PyTorch-like models."""

    def forward(self, *args: Any, **kwargs: Any):  # pragma: no cover - protocol def
        ...


class ModelFactory:
    """Create model instances based on :class:`ModelConfig` values."""

    @staticmethod
    def create(
        config: ModelConfig,
        *,
        embedding_matrix: Any | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> SupportsForward:
        """Instantiate a model described by ``config``.

        Args:
            config: Configuration describing the desired model.
            embedding_matrix: Optional pre-trained embedding weights to use
                when constructing LSTM based architectures.
            extra_kwargs: Additional keyword arguments forwarded to the
                underlying model constructors. This enables extension points
                such as custom output dimensions without modifying the
                configuration schema.

        Returns:
            A model instance matching ``config``.
        """

        config.validate()
        kwargs: dict[str, Any] = dict(extra_kwargs or {})
        if config.architecture == "lstm":
            if EntropyLSTM is None:
                raise RuntimeError("EntropyLSTM requires torch to be installed")
            return EntropyLSTM(
                vocab_size=config.vocab_size,
                embed_dim=config.embed_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                embedding_matrix=embedding_matrix,
                **kwargs,
            )
        if config.architecture == "lstm_attention":
            if EntropyLSTMAttention is None:
                raise RuntimeError(
                    "EntropyLSTMAttention requires torch to be installed"
                )
            return EntropyLSTMAttention(
                vocab_size=config.vocab_size,
                embed_dim=config.embed_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                num_heads=config.num_heads,
                embedding_matrix=embedding_matrix,
                **kwargs,
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
                **kwargs,
            )
        raise ValueError(f"Unknown architecture: {config.architecture}")

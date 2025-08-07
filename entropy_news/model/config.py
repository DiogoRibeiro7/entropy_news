from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    """Configuration parameters for model creation.

    This dataclass encapsulates common model hyperparameters and
    provides basic validation and serialization helpers. The
    ``architecture`` field determines which model type to build.
    Currently only ``"lstm"`` is supported.
    """

    architecture: str
    vocab_size: int
    embed_dim: int = 100
    hidden_dim: int = 16
    num_layers: int = 1
    dropout: float = 0.0

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any parameter is outside the valid range or
                if the architecture is unsupported.
        """
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.architecture not in {"lstm"}:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

    def to_dict(self) -> dict[str, int | float | str]:
        """Serialize configuration to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, int | float | str]) -> ModelConfig:
        """Create a configuration instance from ``data``."""
        return cls(**data)

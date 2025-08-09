from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration parameters for model creation.

    This dataclass encapsulates common model hyperparameters and
    provides basic validation and serialization helpers. The
    ``architecture`` field determines which model type to build.
    Supported values are ``"lstm"``, ``"lstm_attention"``, and
    ``"transformer"``.
    """

    architecture: str
    vocab_size: int
    embed_dim: int = 100
    hidden_dim: int = 16
    num_heads: int = 2
    ff_dim: int = 128
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
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.ff_dim <= 0:
            raise ValueError("ff_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.architecture not in {"lstm", "lstm_attention", "transformer"}:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        if self.architecture == "transformer" and self.embed_dim % self.num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads for transformer"
            )
        if self.architecture == "lstm_attention" and self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads for lstm_attention"
            )

    def to_dict(self) -> dict[str, int | float | str]:
        """Serialize configuration to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, int | float | str]) -> ModelConfig:
        """Create a configuration instance from ``data``."""
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def to_json(self) -> str:
        """Serialize configuration to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> ModelConfig:
        """Deserialize configuration from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Write configuration to ``path`` in JSON format."""
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        """Load configuration from ``path`` and validate it."""
        return cls.from_json(Path(path).read_text())

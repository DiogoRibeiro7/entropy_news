from .config import ModelConfig
from .factory import ModelFactory

try:  # Optional torch dependency
    from .lstm_entropy import EntropyLSTM
    from .transformer_entropy import EntropyTransformer
    from .trainer import Trainer
except Exception:  # pragma: no cover - torch missing
    EntropyLSTM = None  # type: ignore
    EntropyTransformer = None  # type: ignore
    Trainer = None  # type: ignore

__all__ = [
    "EntropyLSTM",
    "EntropyTransformer",
    "Trainer",
    "ModelConfig",
    "ModelFactory",
]

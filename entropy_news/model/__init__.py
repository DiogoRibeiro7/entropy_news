from .config import ModelConfig
from .factory import ModelFactory

try:  # Optional torch dependency
    from .lstm_entropy import EntropyLSTM
    from .lstm_attention import EntropyLSTMAttention
    from .transformer_entropy import EntropyTransformer
    from .trainer import Trainer
except Exception:  # pragma: no cover - torch missing
    EntropyLSTM = None  # type: ignore
    EntropyLSTMAttention = None  # type: ignore
    EntropyTransformer = None  # type: ignore
    Trainer = None  # type: ignore

__all__ = [
    "EntropyLSTM",
    "EntropyLSTMAttention",
    "EntropyTransformer",
    "Trainer",
    "ModelConfig",
    "ModelFactory",
]

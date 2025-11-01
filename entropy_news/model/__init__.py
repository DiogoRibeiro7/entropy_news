from .config import ModelConfig
from .factory import ModelFactory
from .fusion import ConcatFusion, WeightedFusion
from .inference import export_to_onnx, quantize_dynamic
from .distributed import (
    CheckpointManager,
    TrainingMetrics,
    init_distributed,
    monitor_training,
    stress_test,
    synchronize_metrics,
)
from .orchestration import (
    ClusterTopology,
    EnterpriseOrchestrator,
    LaunchSpec,
    NodeConfig,
    TrainingJob,
)

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
    "ConcatFusion",
    "WeightedFusion",
    "quantize_dynamic",
    "export_to_onnx",
    "init_distributed",
    "synchronize_metrics",
    "monitor_training",
    "CheckpointManager",
    "stress_test",
    "TrainingMetrics",
    "EnterpriseOrchestrator",
    "ClusterTopology",
    "NodeConfig",
    "TrainingJob",
    "LaunchSpec",
]

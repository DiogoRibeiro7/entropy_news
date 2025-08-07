"""Shared type aliases for Entropy News."""

from typing import Any, TypeAlias

try:  # Optional numpy dependency for type checking
    import numpy as np
    EmbeddingMatrix: TypeAlias = np.ndarray
except ModuleNotFoundError:  # pragma: no cover - fallback when numpy missing
    EmbeddingMatrix: TypeAlias = Any

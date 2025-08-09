"""Layers for combining text and market features."""

from __future__ import annotations

from typing import Any, List, Sequence

try:  # pragma: no cover - optional dependency
    import torch
    from torch import Tensor, nn
except Exception:  # pragma: no cover - fallback when torch is missing
    torch = None
    Tensor = Any
    nn = object  # type: ignore[misc]


if torch:
    class ConcatFusion(nn.Module):  # pragma: no cover - executed only with torch
        """Concatenate text and market tensors."""

        def forward(self, text: Tensor, market: Tensor) -> Tensor:
            """Return ``text`` concatenated with ``market`` along the last dimension."""
            return torch.cat((text, market), dim=-1)
else:
    class ConcatFusion:
        """Concatenate text and market feature sequences."""

        def __call__(self, text: Sequence[float], market: Sequence[float]) -> List[float]:
            """Return ``text`` concatenated with ``market`` as a Python list."""
            return list(text) + list(market)

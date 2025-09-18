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


    class WeightedFusion(nn.Module):  # pragma: no cover - executed only with torch
        """Blend modalities using learnable weights."""

        def __init__(
            self,
            *,
            init_weights: tuple[float, float] = (0.5, 0.5),
            learnable: bool = True,
        ) -> None:
            super().__init__()
            weights = torch.tensor(init_weights, dtype=torch.float32)
            if learnable:
                self.logits = nn.Parameter(torch.log(weights))
            else:
                self.register_buffer("_weights", weights)

        @property
        def weights(self) -> Tensor:
            """Return the current weight distribution."""

            if hasattr(self, "logits"):
                return torch.softmax(self.logits, dim=0)
            return getattr(self, "_weights")

        def forward(self, text: Tensor, market: Tensor) -> Tensor:
            """Compute weighted combination of ``text`` and ``market`` signals."""

            w_text, w_market = self.weights
            return w_text * text + w_market * market
else:
    class ConcatFusion:
        """Concatenate text and market feature sequences."""

        def __call__(self, text: Sequence[float], market: Sequence[float]) -> List[float]:
            """Return ``text`` concatenated with ``market`` as a Python list."""
            return list(text) + list(market)


    class WeightedFusion:
        """Blend modalities using scalar weights."""

        def __init__(self, *, init_weights: tuple[float, float] = (0.5, 0.5)) -> None:
            self.weights = init_weights

        def __call__(self, text: Sequence[float], market: Sequence[float]) -> List[float]:
            w_text, w_market = self.weights
            return [w_text * t + w_market * m for t, m in zip(text, market)]

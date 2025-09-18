"""Metrics for analysing multi-modal fusion behaviour."""

from __future__ import annotations

from typing import Iterable


def modality_contribution(weights: Iterable[float]) -> dict[str, float]:
    """Normalize ``weights`` into a contribution dictionary."""

    values = [max(float(w), 0.0) for w in weights]
    if len(values) != 2:
        raise ValueError("Expected exactly two weights: text and market")
    total = sum(values)
    if total == 0:
        return {"text": 0.5, "market": 0.5}
    text_weight, market_weight = values
    return {"text": text_weight / total, "market": market_weight / total}


def balance_score(weights: Iterable[float]) -> float:
    """Return a score in ``[0, 1]`` indicating how balanced the weights are."""

    contrib = modality_contribution(weights)
    return 1.0 - abs(contrib["text"] - contrib["market"])


def alignment_score(
    text_signal: Iterable[float],
    market_signal: Iterable[float],
    weights: Iterable[float],
) -> float:
    """Compare observed signal strengths with the configured weights."""

    text_strength = sum(abs(float(v)) for v in text_signal)
    market_strength = sum(abs(float(v)) for v in market_signal)
    observed = modality_contribution([text_strength, market_strength])
    target = modality_contribution(weights)
    return 1.0 - abs(observed["text"] - target["text"])

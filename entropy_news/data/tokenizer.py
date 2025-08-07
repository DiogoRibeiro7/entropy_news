"""Tokenization strategies used by :mod:`entropy_news` preprocessing."""

from __future__ import annotations

from typing import List, Protocol


class Tokenizer(Protocol):
    """Protocol for tokenization strategies."""

    def tokenize(self, text: str) -> List[str]:  # pragma: no cover - interface
        """Split ``text`` into tokens."""


class WhitespaceTokenizer:
    """Tokenize text by splitting on whitespace."""

    def tokenize(self, text: str) -> List[str]:
        return text.split()


__all__ = ["Tokenizer", "WhitespaceTokenizer"]


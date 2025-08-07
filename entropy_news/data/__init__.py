"""Data preprocessing utilities."""

from .preprocessor import TextPreprocessor
from .tokenizer import Tokenizer, WhitespaceTokenizer

__all__ = ["TextPreprocessor", "Tokenizer", "WhitespaceTokenizer"]

try:  # Optional torch-dependent datasets
    from .dataset import NewsDataset
    from .streaming_dataset import StreamingNewsDataset

    __all__ += ["NewsDataset", "StreamingNewsDataset"]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pass

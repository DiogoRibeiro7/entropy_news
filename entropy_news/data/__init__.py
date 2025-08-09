"""Data preprocessing utilities."""

from .preprocessor import TextPreprocessor
from .tokenizer import Tokenizer, WhitespaceTokenizer
from .market import MarketRecord, load_market_csv

__all__ = [
    "TextPreprocessor",
    "Tokenizer",
    "WhitespaceTokenizer",
    "MarketRecord",
    "load_market_csv",
]

try:  # Optional torch-dependent datasets
    from .dataset import NewsDataset
    from .streaming_dataset import StreamingNewsDataset

    __all__ += ["NewsDataset", "StreamingNewsDataset"]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pass

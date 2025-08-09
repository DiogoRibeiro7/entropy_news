"""Data preprocessing utilities."""

from .preprocessor import TextPreprocessor
from .tokenizer import Tokenizer, WhitespaceTokenizer
from .market import (
    MarketRecord,
    load_market_csv,
    fetch_yahoo_history,
    fetch_alpha_vantage_history,
)

__all__ = [
    "TextPreprocessor",
    "Tokenizer",
    "WhitespaceTokenizer",
    "MarketRecord",
    "load_market_csv",
    "fetch_yahoo_history",
    "fetch_alpha_vantage_history",
]

try:  # Optional torch-dependent datasets
    from .dataset import NewsDataset
    from .streaming_dataset import StreamingNewsDataset

    __all__ += ["NewsDataset", "StreamingNewsDataset"]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pass

"""Top level package for :mod:`entropy_news`."""

from typing import TYPE_CHECKING

from .utils.logger import setup_logger
from .utils.metrics import perplexity

__all__ = [
    "TextPreprocessor",
    "NewsDataset",
    "EntropyLSTM",
    "Trainer",
    "EntropyCalculator",
    "NewsModelUpdateCalculator",
    "setup_logger",
    "perplexity",
]

if TYPE_CHECKING:  # pragma: no cover - help type checkers only
    from .data.preprocessor import TextPreprocessor
    from .data.dataset import NewsDataset
    from .model.lstm_entropy import EntropyLSTM
    from .model.trainer import Trainer
    from .evaluation.entropy_calculator import EntropyCalculator
    from .evaluation.news_model_update import NewsModelUpdateCalculator


def __getattr__(name: str) -> object:
    """Dynamically import and return top-level package attributes.

    Args:
        name: Requested attribute name.

    Returns:
        Imported object corresponding to ``name``.

    Raises:
        AttributeError: If ``name`` is not a valid top-level attribute.
    """
    if name == "TextPreprocessor" or name == "NewsDataset":
        from .data import preprocessor, dataset

        return getattr(preprocessor if name == "TextPreprocessor" else dataset, name)
    if name == "EntropyLSTM" or name == "Trainer":
        from .model import lstm_entropy, trainer

        return getattr(lstm_entropy if name == "EntropyLSTM" else trainer, name)
    if name == "EntropyCalculator" or name == "NewsModelUpdateCalculator":
        from .evaluation import entropy_calculator, news_model_update

        return getattr(
            entropy_calculator if name == "EntropyCalculator" else news_model_update,
            name,
        )
    if name == "setup_logger" or name == "perplexity":
        from .utils import logger, metrics

        return getattr(logger if name == "setup_logger" else metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

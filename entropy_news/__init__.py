"""Top level package for :mod:`entropy_news`."""

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

def __getattr__(name):
    if name == "TextPreprocessor" or name == "NewsDataset":
        from .data import preprocessor, dataset
        return getattr(preprocessor if name == "TextPreprocessor" else dataset, name)
    if name == "EntropyLSTM" or name == "Trainer":
        from .model import lstm_entropy, trainer
        return getattr(lstm_entropy if name == "EntropyLSTM" else trainer, name)
    if name == "EntropyCalculator" or name == "NewsModelUpdateCalculator":
        from .evaluation import entropy_calculator, news_model_update
        return getattr(entropy_calculator if name == "EntropyCalculator" else news_model_update, name)
    if name == "setup_logger" or name == "perplexity":
        from .utils import logger, metrics
        return getattr(logger if name == "setup_logger" else metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# entropy_news/rolling_train_forecast.py
from __future__ import annotations

"""Rolling window pipeline utilities."""

import argparse
import inspect
import os
import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from entropy_news.utils import (
    ConfigDefaults,
    ConfigOverrides,
    get_device,
    load_base_config,
    load_texts,
    resolve_model_config,
    setup_logger,
)

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from torch import nn

    from entropy_news.data import NewsDataset, TextPreprocessor
    from entropy_news.model import ModelConfig
    from entropy_news.model.factory import SupportsForward
logger = logging.getLogger("train_logger")


def prepare_training_set(
    months: list[str],
    base_data_dir: str,
    seq_len: int,
    vocab_size: int,
    lazy: bool = False,
) -> tuple[NewsDataset, TextPreprocessor]:
    """Create a dataset and preprocessor from historical months.

    Args:
        months: Ordered list of months used for training.
        base_data_dir: Directory containing ``news_<month>.txt`` files.
        seq_len: Maximum sequence length for the dataset.
        vocab_size: Number of words to keep in the vocabulary.
        lazy: Whether to lazily pad sequences to reduce memory usage.

    Returns:
        Tuple with the prepared dataset and the fitted preprocessor.
    """

    from entropy_news.data import NewsDataset, TextPreprocessor

    texts = load_texts_for_months(months, base_data_dir)
    preprocessor = TextPreprocessor(vocab_size=vocab_size)
    preprocessor.build_vocab(texts)
    encoded = [preprocessor.encode(t) for t in texts]
    dataset = NewsDataset(
        encoded, seq_len=seq_len, in_memory=not lazy
    )
    # Return both the processed dataset and the fitted preprocessor
    return dataset, preprocessor


def train_model(
    dataset: "NewsDataset",
    preprocessor: "TextPreprocessor",
    config: "ModelConfig",
    learning_rate: float,
    epochs: int,
    batch_size: int,
    device: "torch.device" | None = None,
    show_progress: bool = True,
) -> "SupportsForward":
    """Train a model defined by ``config`` on ``dataset``.

    Args:
        dataset: Training dataset.
        preprocessor: Preprocessor fitted on the training window.
        config: Model configuration describing the architecture.
        learning_rate: Optimiser learning rate.
        epochs: Number of training epochs.
        batch_size: Samples per batch.
        device: Optional ``torch`` device for computation.
        show_progress: Whether to display a progress bar during training.

    Returns:
        The trained model instance produced by :class:`ModelFactory`.
    """

    from entropy_news.model import ModelFactory, Trainer

    device = device or get_device()
    effective_config = replace(config, vocab_size=len(preprocessor.vocab))
    model = ModelFactory.create(
        effective_config,
        embedding_matrix=preprocessor.embedding_matrix,
    ).to(device)
    trainer = Trainer(model, learning_rate=learning_rate, device=device)
    trainer.train(
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    return model


def update_with_new_month(
    model: "SupportsForward",
    preprocessor: "TextPreprocessor",
    new_texts: list[str],
    seq_len: int,
    config: "ModelConfig",
    fine_tune_epochs: int,
    learning_rate: float,
    *,
    device: "torch.device" | None = None,
    lazy: bool = False,
    show_progress: bool = True,
    batch_size: int = 32,
) -> dict[str, float]:
    """Fine-tune ``model`` on new data and compute entropies.

    Args:
        model: Base model to be updated.
        preprocessor: Preprocessor fitted on the training window.
        new_texts: List of raw news strings for the new month.
        seq_len: Sequence length for the fine-tuning dataset.
        config: Model configuration describing architecture hyperparameters.
        fine_tune_epochs: Number of fine-tuning epochs.
        learning_rate: Optimiser learning rate.
        device: Optional ``torch`` device for computation.
        lazy: Whether to lazily pad the dataset.
        show_progress: Whether to display progress bars for training and entropy computation.
        batch_size: Mini-batch size used during fine-tuning.

    Returns:
        Entropy metrics for the updated model.
    """

    from entropy_news.data import NewsDataset
    from entropy_news.evaluation import NewsModelUpdateCalculator
    from entropy_news.model import ModelFactory, Trainer

    device = device or get_device()

    encoded_new = [preprocessor.encode(t) for t in new_texts]
    new_dataset = NewsDataset(
        encoded_new, seq_len=seq_len, in_memory=not lazy
    )

    effective_config = replace(config, vocab_size=len(preprocessor.vocab))
    model_old = ModelFactory.create(
        effective_config,
        embedding_matrix=preprocessor.embedding_matrix,
    ).to(device)
    model_old.load_state_dict(model.state_dict())

    trainer = Trainer(model, learning_rate=learning_rate, device=device)
    trainer.fine_tune(
        new_dataset,
        epochs=fine_tune_epochs,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    calculator = NewsModelUpdateCalculator(model_old, model, device=device)
    return calculator.compute_entropies(
        new_dataset, show_progress=show_progress
    )


def rolling_pipeline(
    months: list[str],
    base_data_dir: str,
    output_dir: str,
    *,
    seq_len: int = 100,
    train_window_size: int = 6,
    vocab_size: int | None = None,
    architecture: str | None = None,
    embed_dim: int | None = None,
    hidden_dim: int | None = None,
    num_heads: int | None = None,
    ff_dim: int | None = None,
    num_layers: int | None = None,
    dropout: float | None = None,
    learning_rate: float = 0.001,
    train_epochs: int = 50,
    batch_size: int = 128,
    fine_tune_epochs: int = 5,
    fine_tune_batch_size: int = 32,
    config: "ModelConfig" | None = None,
    lazy: bool = False,
    show_progress: bool = True,
) -> None:
    """Run a rolling training and forecasting pipeline."""

    import pandas as pd
    from entropy_news.model import ModelConfig

    resolved_config = resolve_model_config(
        base_config=config,
        overrides=ConfigOverrides(
            architecture=architecture,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=dropout,
        ),
        defaults=ConfigDefaults(),
    )
    resolved_config.validate()

    device = get_device()
    os.makedirs(output_dir, exist_ok=True)
    results: list[dict[str, float]] = []

    for idx in range(train_window_size, len(months)):
        current_month = months[idx]
        logger.info("Processing month: %s", current_month)

        train_months = months[idx - train_window_size : idx]
        signature = inspect.signature(prepare_training_set)
        prepare_args = (
            train_months,
            base_data_dir,
            seq_len,
            resolved_config.vocab_size,
        )
        if "lazy" in signature.parameters:
            train_dataset, preprocessor = prepare_training_set(*prepare_args, lazy=lazy)
        else:
            train_dataset, preprocessor = prepare_training_set(*prepare_args)

        model = train_model(
            train_dataset,
            preprocessor,
            resolved_config,
            learning_rate=learning_rate,
            epochs=train_epochs,
            batch_size=batch_size,
            device=device,
            show_progress=show_progress,
        )

        new_texts = load_texts_for_month(current_month, base_data_dir)
        entropies = update_with_new_month(
            model,
            preprocessor,
            new_texts,
            seq_len=seq_len,
            config=resolved_config,
            fine_tune_epochs=fine_tune_epochs,
            learning_rate=learning_rate,
            device=device,
            lazy=lazy,
            show_progress=show_progress,
            batch_size=fine_tune_batch_size,
        )
        entropies["month"] = current_month
        results.append(entropies)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "rolling_forecast_results.csv"), index=False)
    logger.info(
        "Rolling forecast results saved to %s/rolling_forecast_results.csv",
        output_dir,
    )

def load_texts_for_month(month: str, base_data_dir: str) -> list[str]:
    """Load texts for a single month.

    Args:
        month: Target month in ``YYYY-MM`` format.
        base_data_dir: Directory containing ``news_<month>.txt`` files.

    Returns:
        List of news strings. Missing files yield an empty list.
    """

    file_path = os.path.join(base_data_dir, f"news_{month}.txt")
    if not os.path.exists(file_path):
        logger.warning("Monthly file missing: %s", file_path)
        return []

    try:
        # Delegate actual reading to ``utils.load_texts``
        return load_texts(file_path)
    except OSError as exc:  # pragma: no cover - unlikely to occur in tests
        logger.error("Failed to read %s: %s", file_path, exc)
        raise

    
def load_texts_for_months(months: list[str], base_data_dir: str) -> list[str]:
    """Load and combine texts from multiple months.

    Args:
        months: Sequence of months in ``YYYY-MM`` format.
        base_data_dir: Directory containing the monthly files.

    Returns:
        Concatenated list of all texts.
    """
    texts: list[str] = []
    for month in months:
        texts += load_texts_for_month(month, base_data_dir)
    return texts

def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run rolling entropy forecasting")
    parser.add_argument("months", nargs="+", help="Ordered list of months to process")
    parser.add_argument("--base-data-dir", default="data/", help="Directory with monthly files")
    parser.add_argument("--output-dir", default="output/", help="Directory for results")
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--train-window-size", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument(
        "--architecture",
        choices=["lstm", "lstm_attention", "transformer"],
        default=None,
        help="Model architecture to use during training",
    )
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--train-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-batch-size", type=int, default=32)
    parser.add_argument(
        "--model-config",
        default=None,
        help="Optional path to a saved ModelConfig JSON",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to a log file; if omitted only console logging is used",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Defer dataset padding to reduce memory usage",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        help="Disable progress bars",
    )
    parser.set_defaults(progress=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Execute the rolling forecast pipeline from the command line."""

    parser = build_parser()
    args = parser.parse_args(argv)

    global logger
    logger = setup_logger("train_logger", args.log_file)

    base_config = None
    if args.model_config:
        try:
            base_config = load_base_config(args.model_config)
        except (OSError, ValueError) as exc:
            logger.error("Failed to load model configuration: %s", exc)
            raise SystemExit(1) from exc

    rolling_pipeline(
        months=args.months,
        base_data_dir=args.base_data_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        train_window_size=args.train_window_size,
        vocab_size=args.vocab_size,
        architecture=args.architecture,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_batch_size=args.fine_tune_batch_size,
        config=base_config,
        lazy=args.lazy,
        show_progress=args.progress,
    )


if __name__ == "__main__":
    main()

# entropy_news/rolling_train_forecast.py
from __future__ import annotations

"""Rolling window pipeline utilities."""

import argparse
import os
import logging
from typing import TYPE_CHECKING

from entropy_news.utils import load_texts, setup_logger, get_device

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from entropy_news.data import NewsDataset, TextPreprocessor
    from entropy_news.evaluation import NewsModelUpdateCalculator
    from entropy_news.model import EntropyLSTM
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

    Returns:
        Tuple with the prepared dataset and the fitted preprocessor.
    """

    from entropy_news.data import NewsDataset, TextPreprocessor

    texts = load_texts_for_months(months, base_data_dir)
    preprocessor = TextPreprocessor(vocab_size=vocab_size)
    preprocessor.build_vocab(texts)
    encoded = [preprocessor.encode(t) for t in texts]
    dataset = NewsDataset(encoded, seq_len=seq_len, lazy=lazy)
    # Return both the processed dataset and the fitted preprocessor
    return dataset, preprocessor


def train_model(
    dataset: NewsDataset,
    vocab_size: int,
    embed_dim: int,
    hidden_dim: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    device: torch.device | None = None,
    show_progress: bool = True,
) -> EntropyLSTM:
    """Train an ``EntropyLSTM`` model.

    Args:
        dataset: Training dataset.
        vocab_size: Vocabulary size for the embedding layer.
        embed_dim: Dimension of the embeddings.
        hidden_dim: Hidden state dimension of the LSTM.
        learning_rate: Optimiser learning rate.
        epochs: Number of training epochs.
        batch_size: Samples per batch.

        device: Optional ``torch`` device for computation.
        show_progress: Whether to display a progress bar during training.

    Returns:
        The trained ``EntropyLSTM`` instance.
    """

    from entropy_news.model import EntropyLSTM, Trainer

    device = device or get_device()
    # Instantiate a fresh model for this training window
    model = EntropyLSTM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    trainer = Trainer(model, learning_rate=learning_rate, device=device)
    trainer.train(
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    # Trained model ready for evaluation
    return model


def update_with_new_month(
    model: EntropyLSTM,
    preprocessor: TextPreprocessor,
    new_texts: list[str],
    seq_len: int,
    embed_dim: int,
    hidden_dim: int,
    fine_tune_epochs: int,
    learning_rate: float,
    device: torch.device | None = None,
    lazy: bool = False,
    show_progress: bool = True,
) -> dict[str, float]:
    """Fine-tune ``model`` on new data and compute entropies.

    Args:
        model: Base model to be updated.
        preprocessor: Preprocessor fitted on the training window.
        new_texts: List of raw news strings for the new month.
        seq_len: Sequence length for the fine-tuning dataset.
        embed_dim: Embedding dimension of the models.
        hidden_dim: Hidden state dimension of the models.
        fine_tune_epochs: Number of fine-tuning epochs.
        learning_rate: Optimiser learning rate.
        device: Optional ``torch`` device for computation.
        lazy: Whether to lazily pad the dataset.
        show_progress: Whether to display progress bars for training and
            entropy computation.

    Returns:
        Entropy metrics for the updated model.
    """

    from entropy_news.data import NewsDataset
    from entropy_news.evaluation import NewsModelUpdateCalculator
    from entropy_news.model import EntropyLSTM, Trainer

    device = device or get_device()

    encoded_new = [preprocessor.encode(t) for t in new_texts]
    # Dataset representing the new month's articles
    new_dataset = NewsDataset(encoded_new, seq_len=seq_len, lazy=lazy)

    # Clone current parameters before fine-tuning
    model_old = EntropyLSTM(
        vocab_size=len(preprocessor.vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    model_old.load_state_dict(model.state_dict())

    trainer = Trainer(model, learning_rate=learning_rate, device=device)
    trainer.fine_tune(
        new_dataset,
        epochs=fine_tune_epochs,
        batch_size=32,
        show_progress=show_progress,
    )

    calculator = NewsModelUpdateCalculator(model_old, model, device=device)
    # Compare old and updated models on the new data
    return calculator.compute_entropies(
        new_dataset, show_progress=show_progress
    )


def rolling_pipeline(
    months: list[str],
    base_data_dir: str,
    output_dir: str,
    seq_len: int = 100,
    train_window_size: int = 6,
    lazy: bool = False,
    show_progress: bool = True,
):
    """Run a rolling training and forecasting pipeline.

    Args:
        months: Sequence of months in ``YYYY-MM`` format.
        base_data_dir: Directory containing ``news_<month>.txt`` files.
        output_dir: Directory where the CSV results are written.
        seq_len: Sequence length used when constructing datasets.
        train_window_size: Number of months used for each training window.
        lazy: Whether to lazily pad datasets.
        show_progress: Whether to display progress bars during training and
            evaluation.

    This function trains a fresh model for each window of ``train_window_size``
    months, evaluates it on the following month and appends the entropies to
    ``rolling_forecast_results.csv``.
    """
    import pandas as pd

    # Hyperparameters
    vocab_size = 10000
    embed_dim = 100
    hidden_dim = 16
    batch_size = 128
    train_epochs = 50
    fine_tune_epochs = 5
    learning_rate = 0.001
    device = get_device()

    os.makedirs(output_dir, exist_ok=True)

    results = []

    # Iterate over each month after the initial training window
    for idx in range(train_window_size, len(months)):
        current_month = months[idx]
        logger.info(f"Processing month: {current_month}")

        train_months = months[idx - train_window_size : idx]
        train_dataset, preprocessor = prepare_training_set(
            train_months, base_data_dir, seq_len, vocab_size, lazy=lazy
        )

        model = train_model(
            train_dataset,
            vocab_size=len(preprocessor.vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            epochs=train_epochs,
            batch_size=batch_size,
            device=device,
            show_progress=show_progress,
        )

        # Load new month's data
        new_texts = load_texts_for_month(current_month, base_data_dir)
        entropies = update_with_new_month(
            model,
            preprocessor,
            new_texts,
            seq_len=seq_len,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            fine_tune_epochs=fine_tune_epochs,
            learning_rate=learning_rate,
            device=device,
            lazy=lazy,
            show_progress=show_progress,
        )
        entropies["month"] = current_month
        results.append(entropies)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "rolling_forecast_results.csv"), index=False)
    logger.info(
        f"Rolling forecast results saved to {output_dir}/rolling_forecast_results.csv"
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
    """Create the CLI argument parser.

    Returns:
        Configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(description="Run rolling entropy forecasting")
    parser.add_argument("months", nargs="+", help="Ordered list of months to process")
    parser.add_argument("--base-data-dir", default="data/", help="Directory with monthly files")
    parser.add_argument("--output-dir", default="output/", help="Directory for results")
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--train-window-size", type=int, default=6)
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
    """Execute the rolling forecast pipeline from the command line.

    Args:
        argv: Optional sequence of command-line arguments.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    global logger
    logger = setup_logger("train_logger", args.log_file)
    rolling_pipeline(
        months=args.months,
        base_data_dir=args.base_data_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        train_window_size=args.train_window_size,
        lazy=args.lazy,
        show_progress=args.progress,
    )


if __name__ == "__main__":
    main()

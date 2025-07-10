# entropy_news/rolling_train_forecast.py

"""Rolling window pipeline utilities."""

import argparse
import os


from entropy_news.utils import load_texts, setup_logger

logger = setup_logger("train_logger", "logs/train.log")


def rolling_pipeline(
    months: list[str],
    base_data_dir: str,
    output_dir: str,
    seq_len: int = 100,
    train_window_size: int = 6,
):
    """Run a rolling training and forecasting pipeline.

    Parameters
    ----------
    months : list[str]
        Ordered list of month identifiers used to locate input files.
    base_data_dir : str
        Directory containing ``news_<month>.txt`` files.
    output_dir : str
        Directory in which the results CSV is written.
    seq_len : int, optional
        Token sequence length for the dataset, by default ``100``.
    train_window_size : int, optional
        Number of months used in the initial training window, by default ``6``.
    """
    import pandas as pd
    from entropy_news.data import TextPreprocessor, NewsDataset
    from entropy_news.model import EntropyLSTM, Trainer
    from entropy_news.evaluation import NewsModelUpdateCalculator

    # Hyperparameters
    vocab_size = 10000
    embed_dim = 100
    hidden_dim = 16
    batch_size = 128
    train_epochs = 50
    fine_tune_epochs = 5
    learning_rate = 0.001

    os.makedirs(output_dir, exist_ok=True)

    results = []

    # Iterate over each month after the initial training window
    for idx in range(train_window_size, len(months)):
        current_month = months[idx]
        logger.info(f"Processing month: {current_month}")

        train_months = months[idx - train_window_size : idx]
        train_texts = load_texts_for_months(train_months, base_data_dir)
        preprocessor = TextPreprocessor(vocab_size=vocab_size)
        preprocessor.build_vocab(train_texts)
        encoded_train = [preprocessor.encode(t) for t in train_texts]
        train_dataset = NewsDataset(encoded_train, seq_len=seq_len)

        model = EntropyLSTM(
            vocab_size=len(preprocessor.vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )
        model = model.to(model.device)
        trainer = Trainer(model, learning_rate=learning_rate)
        trainer.train(train_dataset, epochs=train_epochs, batch_size=batch_size)

        # Load new month's data
        new_texts = load_texts_for_month(current_month, base_data_dir)
        encoded_new = [preprocessor.encode(t) for t in new_texts]
        new_dataset = NewsDataset(encoded_new, seq_len=seq_len)

        model_old = EntropyLSTM(
            vocab_size=len(preprocessor.vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )
        model_old.load_state_dict(model.state_dict())
        model_old = model_old.to(model_old.device)

        trainer = Trainer(model)
        trainer.fine_tune(new_dataset, epochs=fine_tune_epochs, batch_size=32)

        calculator = NewsModelUpdateCalculator(model_old, model)
        entropies = calculator.compute_entropies(new_dataset)
        entropies["month"] = current_month
        results.append(entropies)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "rolling_forecast_results.csv"), index=False)
    logger.info(f"Rolling forecast results saved to {output_dir}/rolling_forecast_results.csv")

def load_texts_for_month(month: str, base_data_dir: str) -> list[str]:
    """Return texts for ``month`` loaded from ``base_data_dir``."""
    file_path = os.path.join(base_data_dir, f"news_{month}.txt")
    return load_texts(file_path)

    
def load_texts_for_months(months: list[str], base_data_dir: str) -> list[str]:
    """Return concatenated texts for ``months`` from ``base_data_dir``."""
    texts: list[str] = []
    for month in months:
        texts += load_texts_for_month(month, base_data_dir)
    return texts

def build_parser() -> argparse.ArgumentParser:
    """Return argument parser for the rolling pipeline CLI."""
    parser = argparse.ArgumentParser(description="Run rolling entropy forecasting")
    parser.add_argument("months", nargs="+", help="Ordered list of months to process")
    parser.add_argument("--base-data-dir", default="data/", help="Directory with monthly files")
    parser.add_argument("--output-dir", default="output/", help="Directory for results")
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--train-window-size", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``entropy-news-rolling`` command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    rolling_pipeline(
        months=args.months,
        base_data_dir=args.base_data_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        train_window_size=args.train_window_size,
    )


if __name__ == "__main__":
    main()

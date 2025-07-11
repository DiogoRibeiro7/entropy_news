# entropy_news/main_forecast.py

import pickle
import argparse
import logging
import os

from entropy_news.utils import setup_logger, load_texts


logger = logging.getLogger("train_logger")


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for the forecasting script.

    Returns:
        Configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(description="Forecast entropies from new data")
    parser.add_argument("--vocab-path", default="output/vocab.pkl", help="Path to saved vocabulary")
    parser.add_argument("--model-path", default="output/model_final.pth", help="Path to trained model")
    parser.add_argument("--new-data", default="data/news_new.txt", help="Text file with new news")
    parser.add_argument("--output-csv", default="output/forecast_results.csv", help="Where to store computed entropies")
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="LSTM dropout between layers")
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=5,
        help="Epochs used when fine-tuning with new data",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to a log file; if omitted only console logging is used",
    )
    return parser

def main(argv: list[str] | None = None) -> None:
    """Run the entropy forecasting workflow.

    Args:
        argv: Optional sequence of command-line arguments.
    """
    import torch
    import pandas as pd
    from entropy_news.data import TextPreprocessor, NewsDataset
    from entropy_news.model import EntropyLSTM, Trainer
    from entropy_news.evaluation import NewsModelUpdateCalculator

    parser = build_parser()
    args = parser.parse_args(argv)

    global logger
    logger = setup_logger("train_logger", args.log_file)

    if not os.path.exists(args.vocab_path):
        logger.error("Vocabulary file not found: %s", args.vocab_path)
        raise SystemExit(1)

    if not os.path.exists(args.model_path):
        logger.error("Model file not found: %s", args.model_path)
        raise SystemExit(1)

    # Load vocabulary with user-friendly error handling
    try:
        with open(args.vocab_path, "rb") as f:
            vocab = pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load vocabulary from %s: %s", args.vocab_path, exc)
        raise SystemExit(1) from exc

    # Preprocess new data
    preprocessor = TextPreprocessor()
    preprocessor.vocab = vocab

    texts = load_texts(args.new_data)
    encoded = [preprocessor.encode(t) for t in texts]
    new_dataset = NewsDataset(encoded, seq_len=args.seq_len)

    # Load previous model
    model_old = EntropyLSTM(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    try:
        state_dict = torch.load(args.model_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load model from %s: %s", args.model_path, exc)
        raise SystemExit(1) from exc
    model_old.load_state_dict(state_dict)
    model_old = model_old.to(model_old.device)

    # Train a new model with the latest data
    model_new = EntropyLSTM(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model_new.load_state_dict(state_dict)
    model_new = model_new.to(model_new.device)

    # Brief fine-tuning to simulate a model update
    from entropy_news.model import Trainer
    trainer = Trainer(model_new)
    trainer.fine_tune(new_dataset, epochs=args.fine_tune_epochs, batch_size=args.batch_size)

    # Calculate ENT, ENT_news and ENT_model
    calculator = NewsModelUpdateCalculator(model_old, model_new)
    entropies = calculator.compute_entropies(new_dataset)

    # Export to CSV
    df = pd.DataFrame([entropies])
    df.to_csv(args.output_csv, index=False)

    logger.info(f"Forecast results exported to {args.output_csv}")


if __name__ == "__main__":
    main()

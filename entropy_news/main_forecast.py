# entropy_news/main_forecast.py

import argparse
import logging
from entropy_news.utils import load_texts, setup_logger
from entropy_news.utils.cli import load_encoded_dataset, load_model_and_vocab


logger = logging.getLogger("train_logger")


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for the forecasting script.

    Returns:
        Configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(description="Forecast entropies from new data")
    parser.add_argument("--vocab-path", default="output/vocab.json", help="Path to saved vocabulary")
    parser.add_argument("--model-path", default="output/model_final.pth", help="Path to trained model")
    parser.add_argument("--config-path", default="output/model_config.json", help="Path to saved model configuration")
    parser.add_argument("--new-data", default="data/news_new.txt", help="Text file with new news")
    parser.add_argument("--output-csv", default="output/forecast_results.csv", help="Where to store computed entropies")
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=None, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=None, help="LSTM dropout between layers")
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
    parser.add_argument(
        "--allow-unsafe-load",
        action="store_true",
        help=(
            "Allow falling back to torch.load(..., weights_only=False) for legacy"
            " checkpoints. Only enable this when the file is trusted."
        ),
    )
    parser.set_defaults(progress=True)
    return parser

def main(argv: list[str] | None = None) -> None:
    """Run the entropy forecasting workflow.

    Args:
        argv: Optional sequence of command-line arguments.
    """
    import pandas as pd
    from entropy_news.evaluation import NewsModelUpdateCalculator
    from entropy_news.model import ModelFactory, Trainer

    parser = build_parser()
    args = parser.parse_args(argv)

    global logger
    logger = setup_logger("train_logger", args.log_file)

    try:
        new_texts = load_texts(args.new_data)
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    try:
        (
            preprocessor,
            model_old,
            device,
            config,
        ) = load_model_and_vocab(
            args.vocab_path,
            args.model_path,
            args.embed_dim,
            args.hidden_dim,
            args.num_layers,
            args.dropout,
            config_path=args.config_path,
            allow_unsafe_load=args.allow_unsafe_load,
        )
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    new_dataset = load_encoded_dataset(
        preprocessor,
        args.new_data,
        seq_len=args.seq_len,
        lazy=args.lazy,
        texts=new_texts,
    )

    # Train a new model with the latest data
    model_new = ModelFactory.create(
        config,
        embedding_matrix=None,
    ).to(device)
    model_new.load_state_dict(model_old.state_dict())

    # Brief fine-tuning to simulate a model update
    trainer = Trainer(model_new, device=device)
    trainer.fine_tune(
        new_dataset,
        epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        show_progress=args.progress,
    )

    # Calculate ENT, ENT_news and ENT_model
    calculator = NewsModelUpdateCalculator(model_old, model_new, device=device)
    entropies = calculator.compute_entropies(
        new_dataset, show_progress=args.progress
    )

    # Export to CSV
    df = pd.DataFrame([entropies])
    df.to_csv(args.output_csv, index=False)

    logger.info(f"Forecast results exported to {args.output_csv}")


if __name__ == "__main__":
    main()

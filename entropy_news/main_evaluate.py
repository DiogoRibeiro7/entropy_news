# entropy_news/main_evaluate.py
"""Command-line script to evaluate a saved model."""

from __future__ import annotations

import argparse
import logging
from entropy_news.utils import load_texts, setup_logger
from entropy_news.utils.cli import load_encoded_dataset, load_model_and_vocab

logger = logging.getLogger("eval_logger")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for model evaluation.

    Returns:
        Configured ``argparse.ArgumentParser`` with all options.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on new text data"
    )
    parser.add_argument(
        "--vocab-path",
        default="output/vocab.json",
        help="Path to saved vocabulary",
    )
    parser.add_argument(
        "--model-path",
        default="output/model_final.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--data",
        default="data/news_new.txt",
        help="Text file to evaluate",
    )
    parser.add_argument(
        "--config-path",
        default="output/model_config.json",
        help="Path to the saved model configuration (JSON)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to store entropy results as CSV",
    )
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=None, help="Number of LSTM layers")
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="LSTM dropout between layers (legacy fallback)",
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
    """Run the evaluation routine.

    Args:
        argv: Optional sequence of command-line arguments.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    global logger
    logger = setup_logger("eval_logger", args.log_file)

    try:
        texts = load_texts(args.data)
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    try:
        preprocessor, model, device, _config = load_model_and_vocab(
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

    dataset = load_encoded_dataset(
        preprocessor,
        args.data,
        seq_len=args.seq_len,
        lazy=args.lazy,
        texts=texts,
    )

    import pandas as pd
    from entropy_news.evaluation import EntropyCalculator

    # Compute entropy and perplexity
    calculator = EntropyCalculator(model, device=device)
    entropy = calculator.compute_entropy(
        dataset, batch_size=args.batch_size, show_progress=args.progress
    )
    perplex = calculator.compute_perplexity(
        dataset, batch_size=args.batch_size, show_progress=args.progress
    )

    results = {"entropy": entropy, "perplexity": perplex}

    if args.output_csv is not None:
        pd.DataFrame([results]).to_csv(args.output_csv, index=False)

    logger.info(f"Entropy: {entropy:.4f}  Perplexity: {perplex:.2f}")


if __name__ == "__main__":
    main()

# entropy_news/main_evaluate.py
"""Command-line script to evaluate a saved model."""

from __future__ import annotations

import argparse
import pickle

import pandas as pd
import torch

from entropy_news.utils import setup_logger, load_texts
from entropy_news.data import TextPreprocessor, NewsDataset
from entropy_news.model import EntropyLSTM
from entropy_news.evaluation import EntropyCalculator

logger = setup_logger("eval_logger", "logs/eval.log")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for model evaluation.

    Returns:
        argparse.ArgumentParser: Configured parser with all supported options.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on new text data"
    )
    parser.add_argument(
        "--vocab-path",
        default="output/vocab.pkl",
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
        "--output-csv",
        default=None,
        help="Optional path to store entropy results as CSV",
    )
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="LSTM dropout between layers")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the evaluation routine.

    Args:
        argv: Optional sequence of command-line arguments.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load vocabulary
    with open(args.vocab_path, "rb") as f:
        vocab: dict[str, int] = pickle.load(f)

    # Preprocess evaluation data
    preprocessor = TextPreprocessor()
    preprocessor.vocab = vocab

    texts = load_texts(args.data)
    encoded = [preprocessor.encode(t) for t in texts]
    dataset = NewsDataset(encoded, seq_len=args.seq_len)

    # Load model
    model = EntropyLSTM(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(model.device)

    # Compute entropy and perplexity
    calculator = EntropyCalculator(model)
    entropy = calculator.compute_entropy(dataset, batch_size=args.batch_size)
    perplex = calculator.compute_perplexity(dataset, batch_size=args.batch_size)

    results = {"entropy": entropy, "perplexity": perplex}

    if args.output_csv is not None:
        pd.DataFrame([results]).to_csv(args.output_csv, index=False)

    logger.info(f"Entropy: {entropy:.4f}  Perplexity: {perplex:.2f}")


if __name__ == "__main__":
    main()

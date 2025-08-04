# entropy_news/main.py

import argparse
import logging

from entropy_news.utils import get_device, load_texts, setup_logger

logger = logging.getLogger("train_logger")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser.

    Returns:
        Configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(description="Train the entropy LSTM model")
    parser.add_argument(
        "--train-data",
        default="data/news_train.txt",
        help="Path to training data text file",
    )
    parser.add_argument(
        "--glove-path", default="glove.6B.100d.txt", help="Path to GloVe embeddings"
    )
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="LSTM dropout between layers")
    parser.add_argument(
        "--model-out",
        default="output/model_final.pth",
        help="File to save the trained model",
    )
    parser.add_argument(
        "--vocab-out", default="output/vocab.json", help="Where to store the vocabulary"
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
        "--checkpoint",
        default=None,
        help="Path to save training checkpoints",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Checkpoint file to resume training from",
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
    """Run the model training workflow.

    Args:
        argv: Optional sequence of command-line arguments.
    """
    import torch
    from entropy_news.data import TextPreprocessor, NewsDataset
    from entropy_news.model import EntropyLSTM, Trainer

    parser = build_parser()
    args = parser.parse_args(argv)

    global logger
    logger = setup_logger("train_logger", args.log_file)

    # Load training data with clearer error reporting
    try:
        texts = load_texts(args.train_data)
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    # Preprocess texts and build vocabulary
    preprocessor = TextPreprocessor(vocab_size=args.vocab_size)
    preprocessor.build_vocab(texts)
    preprocessor.load_glove_embeddings(
        args.glove_path, args.embed_dim, show_progress=args.progress
    )

    encoded = [preprocessor.encode(t) for t in texts]
    dataset = NewsDataset(encoded, seq_len=args.seq_len, lazy=args.lazy)

    device = get_device()
    # Configure the LSTM model
    model = EntropyLSTM(
        vocab_size=len(preprocessor.vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        embedding_matrix=preprocessor.embedding_matrix,
    ).to(device)

    # Train the model
    trainer = Trainer(model, learning_rate=args.learning_rate, device=device)
    start_epoch = 0
    if args.resume_from:
        try:
            start_epoch = trainer.load_checkpoint(args.resume_from)
            logger.info("Resuming training from epoch %s", start_epoch)
        except OSError as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from exc
    trainer.train(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        start_epoch=start_epoch,
        checkpoint_path=args.checkpoint,
        show_progress=args.progress,
    )

    # Save the model
    torch.save(model.state_dict(), args.model_out)
    preprocessor.save_vocab(args.vocab_out)

    logger.info("Training complete and model saved.")


if __name__ == "__main__":
    main()

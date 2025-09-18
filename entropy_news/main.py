# entropy_news/main.py

import argparse
import logging

from entropy_news.utils import (
    ConfigDefaults,
    ConfigOverrides,
    get_device,
    load_base_config,
    load_texts,
    resolve_model_config,
    setup_logger,
)

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
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocabulary size limit (defaults to 10000 when unspecified)",
    )
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument(
        "--architecture",
        choices=["lstm", "lstm_attention", "transformer"],
        default=None,
        help="Model architecture to train (defaults to LSTM when unspecified)",
    )
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of recurrent/transformer layers (defaults vary per model)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout applied between layers (defaults vary per model)",
    )
    parser.add_argument(
        "--model-out",
        default="output/model_final.pth",
        help="File to save the trained model",
    )
    parser.add_argument(
        "--vocab-out", default="output/vocab.json", help="Where to store the vocabulary"
    )
    parser.add_argument(
        "--config-out",
        default="output/model_config.json",
        help="Where to store the resolved model configuration",
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
        "--model-config",
        default=None,
        help="Optional JSON config to load model hyperparameters from",
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
    from dataclasses import replace
    from entropy_news.data import TextPreprocessor, NewsDataset
    from entropy_news.model import ModelConfig, ModelFactory, Trainer

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

    # Determine configuration defaults and preprocess texts
    base_config: ModelConfig | None = None
    if args.model_config:
        try:
            base_config = load_base_config(args.model_config)
        except (OSError, ValueError) as exc:
            logger.error("Failed to load model configuration: %s", exc)
            raise SystemExit(1) from exc

    initial_config = resolve_model_config(
        base_config=base_config,
        overrides=ConfigOverrides(
            architecture=args.architecture,
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ),
        defaults=ConfigDefaults(),
    )

    vocab_limit = initial_config.vocab_size

    # Preprocess texts and build vocabulary
    preprocessor = TextPreprocessor(vocab_size=vocab_limit)
    preprocessor.build_vocab(texts)

    try:
        preprocessor.load_glove_embeddings(
            args.glove_path,
            initial_config.embed_dim,
            show_progress=args.progress,
        )
    except FileNotFoundError as exc:
        logger.error("GloVe embeddings missing: %s", exc)
        raise SystemExit(1) from exc
    except ValueError as exc:
        logger.error("Failed to load GloVe embeddings: %s", exc)
        raise SystemExit(1) from exc

    encoded = [preprocessor.encode(t) for t in texts]
    dataset = NewsDataset(
        encoded, seq_len=args.seq_len, in_memory=not args.lazy
    )

    device = get_device()
    vocab_size = len(preprocessor.vocab)
    config = replace(initial_config, vocab_size=vocab_size)
    config.validate()

    # Configure the model through the factory to support multiple architectures
    model = ModelFactory.create(
        config,
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

    # Save the model and vocabulary, ensuring directories exist
    from pathlib import Path

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    vocab_path = Path(args.vocab_out)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor.save_vocab(str(vocab_path))

    config_path = Path(args.config_out)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(config_path)

    logger.info("Training complete and model saved.")


if __name__ == "__main__":
    main()

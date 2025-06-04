# entropy_news/main.py

import torch
import pickle
import argparse

from entropy_news.utils import setup_logger, load_texts
from entropy_news.data import TextPreprocessor, NewsDataset
from entropy_news.model import EntropyLSTM, Trainer

logger = setup_logger("train_logger", "logs/train.log")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser with defaults matching the old behavior."""
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
        "--vocab-out", default="output/vocab.pkl", help="Where to store the vocabulary"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the training script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Carregar dados
    texts = load_texts(args.train_data)

    # Preprocessar
    preprocessor = TextPreprocessor(vocab_size=args.vocab_size)
    preprocessor.build_vocab(texts)
    preprocessor.load_glove_embeddings(args.glove_path, args.embed_dim)

    encoded = [preprocessor.encode(t) for t in texts]
    dataset = NewsDataset(encoded, seq_len=args.seq_len)

    # Modelo
    model = EntropyLSTM(
        vocab_size=len(preprocessor.vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        embedding_matrix=preprocessor.embedding_matrix,
    )
    model = model.to(model.device)

    # Treinamento
    trainer = Trainer(model, learning_rate=args.learning_rate)
    trainer.train(dataset, epochs=args.epochs, batch_size=args.batch_size)

    # Salvar modelo
    torch.save(model.state_dict(), args.model_out)
    with open(args.vocab_out, "wb") as f:
        pickle.dump(preprocessor.vocab, f)

    logger.info("Treinamento completo e modelo salvo.")


if __name__ == "__main__":
    main()

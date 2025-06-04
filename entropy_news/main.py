# entropy_news/main.py

import torch
import pickle

from entropy_news.utils import setup_logger, load_texts
from entropy_news.data import TextPreprocessor, NewsDataset
from entropy_news.model import EntropyLSTM, Trainer

logger = setup_logger('train_logger', 'logs/train.log')


def main():
    # Parâmetros
    vocab_size = 10000
    seq_len = 100
    embed_dim = 100
    hidden_dim = 16
    batch_size = 128
    epochs = 50
    learning_rate = 0.001
    glove_path = "glove.6B.100d.txt"  # Atualizar para o caminho real

    # Carregar dados
    texts = load_texts("data/news_train.txt")  # Atualizar para o caminho real

    # Preprocessar
    preprocessor = TextPreprocessor(vocab_size=vocab_size)
    preprocessor.build_vocab(texts)
    preprocessor.load_glove_embeddings(glove_path)

    encoded = [preprocessor.encode(t) for t in texts]
    dataset = NewsDataset(encoded, seq_len=seq_len)

    # Modelo
    model = EntropyLSTM(
        vocab_size=len(preprocessor.vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        embedding_matrix=preprocessor.embedding_matrix
    )
    model = model.to(model.device)

    # Treinamento
    trainer = Trainer(model, learning_rate=learning_rate)
    trainer.train(dataset, epochs=epochs, batch_size=batch_size)

    # Salvar modelo
    torch.save(model.state_dict(), "output/model_final.pth")
    with open("output/vocab.pkl", "wb") as f:
        pickle.dump(preprocessor.vocab, f)

    logger.info("Treinamento completo e modelo salvo.")


if __name__ == "__main__":
    main()
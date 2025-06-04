# entropy_news/main_forecast.py

import torch
import pickle
import pandas as pd

from entropy_news.utils import setup_logger, load_texts
from entropy_news.data import TextPreprocessor, NewsDataset
from entropy_news.model import EntropyLSTM
from entropy_news.evaluation import NewsModelUpdateCalculator

logger = setup_logger('train_logger', 'logs/train.log')

def main():
    # Parâmetros
    seq_len = 100
    embed_dim = 100
    hidden_dim = 16
    batch_size = 1

    # Caminhos
    vocab_path = "output/vocab.pkl"
    model_path = "output/model_final.pth"
    new_data_path = "data/news_new.txt"  # Atualizar para o caminho real
    output_csv_path = "output/forecast_results.csv"

    # Carregar vocabulário
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Preprocessar novos dados
    preprocessor = TextPreprocessor()
    preprocessor.vocab = vocab

    texts = load_texts(new_data_path)
    encoded = [preprocessor.encode(t) for t in texts]
    new_dataset = NewsDataset(encoded, seq_len=seq_len)

    # Carregar modelo antigo
    model_old = EntropyLSTM(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim
    )
    model_old.load_state_dict(torch.load(model_path))
    model_old = model_old.to(model_old.device)

    # Treinar novo modelo com novos dados
    model_new = EntropyLSTM(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim
    )
    model_new.load_state_dict(torch.load(model_path))
    model_new = model_new.to(model_new.device)

    # Pequeno fine-tuning para simular atualização
    from entropy_news.model import Trainer
    trainer = Trainer(model_new)
    trainer.fine_tune(new_dataset, epochs=5, batch_size=32)

    # Calcular ENT, ENT_news e ENT_model
    calculator = NewsModelUpdateCalculator(model_old, model_new)
    entropies = calculator.compute_entropies(new_dataset)

    # Exportar para CSV
    df = pd.DataFrame([entropies])
    df.to_csv(output_csv_path, index=False)

    logger.info(f"Resultados de forecast exportados para {output_csv_path}")


if __name__ == "__main__":
    main()

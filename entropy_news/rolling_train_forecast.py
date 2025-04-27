# entropy_news/rolling_train_forecast.py

import logging
import os
import pickle
import pandas as pd
import torch
from data.preprocessor import TextPreprocessor
from data.dataset import NewsDataset
from model.lstm_entropy import EntropyLSTM
from model.trainer import Trainer
from evaluation.news_model_update import NewsModelUpdateCalculator
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rolling_pipeline(months: list, base_data_dir: str, output_dir: str, seq_len: int = 100):
    # Hyperparameters
    vocab_size = 10000
    embed_dim = 100
    hidden_dim = 16
    batch_size = 128
    train_epochs = 50
    fine_tune_epochs = 5
    learning_rate = 0.001

    os.makedirs(output_dir, exist_ok=True)

    # Initialize preprocessor and vocab
    train_texts = load_texts_for_months(months[:-1], base_data_dir)
    preprocessor = TextPreprocessor(vocab_size=vocab_size)
    preprocessor.build_vocab(train_texts)

    encoded = [preprocessor.encode(t) for t in train_texts]
    train_dataset = NewsDataset(encoded, seq_len=seq_len)

    # Initialize and train initial model
    model = EntropyLSTM(
        vocab_size=len(preprocessor.vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim
    )
    model = model.to(model.device)
    trainer = Trainer(model, learning_rate=learning_rate)
    trainer.train(train_dataset, epochs=train_epochs, batch_size=batch_size)

    results = []

    for current_month in months[-1:]:
        logger.info(f"Processing month: {current_month}")

        # Load new month's data
        new_texts = load_texts_for_month(current_month, base_data_dir)
        encoded_new = [preprocessor.encode(t) for t in new_texts]
        new_dataset = NewsDataset(encoded_new, seq_len=seq_len)

        # Clone old model for comparison
        model_old = EntropyLSTM(
            vocab_size=len(preprocessor.vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim
        )
        model_old.load_state_dict(model.state_dict())
        model_old = model_old.to(model_old.device)

        # Fine-tune model with new data
        trainer = Trainer(model)
        trainer.fine_tune(new_dataset, epochs=fine_tune_epochs, batch_size=32)

        # Compute ENT, ENT_news, ENT_model
        calculator = NewsModelUpdateCalculator(model_old, model)
        entropies = calculator.compute_entropies(None, new_dataset)
        entropies['month'] = current_month
        results.append(entropies)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "rolling_forecast_results.csv"), index=False)
    logger.info(f"Rolling forecast results saved to {output_dir}/rolling_forecast_results.csv")

def load_texts_for_month(month: str, base_data_dir: str):
    file_path = os.path.join(base_data_dir, f"news_{month}.txt")
    return load_texts(file_path)

def load_texts(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    # Example usage
    months = ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06", "2023-07"]
    base_data_dir = "data/"
    output_dir = "output/"
    rolling_pipeline(months, base_data_dir, output_dir)

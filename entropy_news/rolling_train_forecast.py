# entropy_news/rolling_train_forecast.py

import os
import pickle

import pandas as pd
import torch

from entropy_news.utils import setup_logger, load_texts
from entropy_news.data import TextPreprocessor, NewsDataset
from entropy_news.model import EntropyLSTM, Trainer
from entropy_news.evaluation import NewsModelUpdateCalculator

logger = setup_logger('train_logger', 'logs/train.log')


def rolling_pipeline(
    months: list,
    base_data_dir: str,
    output_dir: str,
    seq_len: int = 100,
    train_window_size: int = 6,
):
    # Hyperparameters
    vocab_size = 10000
    embed_dim = 100
    hidden_dim = 16
    batch_size = 128
    train_epochs = 50
    fine_tune_epochs = 5
    learning_rate = 0.001

    os.makedirs(output_dir, exist_ok=True)

    # Initialize preprocessor and vocab using the initial training window
    train_texts = load_texts_for_months(months[:train_window_size], base_data_dir)
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

    # Iterate over each month after the initial training window
    for current_month in months[train_window_size:]:
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

        # Compute ENT, ENT_news and ENT_model
        calculator = NewsModelUpdateCalculator(model_old, model)
        entropies = calculator.compute_entropies(new_dataset)
        entropies['month'] = current_month
        results.append(entropies)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "rolling_forecast_results.csv"), index=False)
    logger.info(f"Rolling forecast results saved to {output_dir}/rolling_forecast_results.csv")

def load_texts_for_month(month: str, base_data_dir: str):
    file_path = os.path.join(base_data_dir, f"news_{month}.txt")
    return load_texts(file_path)

    
def load_texts_for_months(months: list, base_data_dir: str):
    texts = []
    for month in months:
        texts += load_texts_for_month(month, base_data_dir)
    return texts

if __name__ == "__main__":
    # Example usage
    months = ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06", "2023-07"]
    base_data_dir = "data/"
    output_dir = "output/"
    train_window_size = 6
    rolling_pipeline(months, base_data_dir, output_dir, train_window_size=train_window_size)

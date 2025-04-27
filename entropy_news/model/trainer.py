# entropy_news/model/trainer.py

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        self.model = model
        self.device = model.device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def train(self, dataset: Dataset, epochs: int = 50, batch_size: int = 128) -> None:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()

        for epoch in range(1, epochs + 1):
            total_loss = 0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = self.criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                logger.info(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    def fine_tune(self, new_dataset: Dataset, epochs: int = 50, batch_size: int = 128) -> None:
        logger.info("Fine-tuning model on new data...")
        self.train(new_dataset, epochs=epochs, batch_size=batch_size)

# Exemplo de uso:
# trainer = Trainer(model)
# trainer.train(dataset)
# trainer.fine_tune(new_dataset)

# entropy_news/model/trainer.py

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Trainer:
    """Utility class to train and fine-tune language models."""

    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        """Initialise with ``model`` and Adam optimiser."""
        self.model = model
        self.device = model.device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def train(
        self,
        dataset: Dataset,
        epochs: int = 50,
        batch_size: int = 128,
        val_dataset: Optional[Dataset] = None,
        early_stopping: bool = False,
        patience: int = 5,
    ) -> None:
        """Train ``self.model`` using ``dataset``.

        Args:
            dataset: Dataset of token sequences.
            epochs: Number of training epochs.
            batch_size: Samples per training batch.
            val_dataset: Optional dataset for validation loss calculation.
            early_stopping: Whether to stop when validation loss stops
                improving.
            patience: Epochs to wait for improvement before stopping.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = (
            DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        )
        self.model.train()

        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            total_loss = 0
            loop = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            for x_batch, y_batch in loop:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), y_batch.view(-1)
                )
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(loader)
            val_loss = (
                self.evaluate(val_loader) if val_loader is not None else None
            )
            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                msg = f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}"
                if val_loss is not None:
                    msg += f" - Val Loss: {val_loss:.4f}"
                logger.info(msg)

            if val_loss is not None and early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} (best val loss {best_val_loss:.4f})"
                    )
                    break

    def fine_tune(self, new_dataset: Dataset, epochs: int = 50, batch_size: int = 128) -> None:
        """Continue training ``self.model`` on ``new_dataset``.

        Args:
            new_dataset: Additional data used for fine-tuning.
            epochs: Number of fine-tuning epochs.
            batch_size: Samples per batch.
        """
        logger.info("Fine-tuning model on new data...")
        self.train(new_dataset, epochs=epochs, batch_size=batch_size)

    def evaluate(self, loader: DataLoader) -> float:
        """Compute average loss on ``loader`` without updating gradients.

        Args:
            loader: Data loader used for evaluation.

        Returns:
            Average loss across all batches.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = self.model(x_batch)
                loss = self.criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                total_loss += loss.item()
        self.model.train()
        return total_loss / len(loader)

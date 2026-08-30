# entropy_news/model/trainer.py

import logging
from pathlib import Path
from time import perf_counter
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from entropy_news.utils import get_device
from entropy_news.utils.metrics import (
    observe_checkpoint,
    observe_training_batch,
    record_gradient_norm,
    record_validation_loss,
    update_training_epoch,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Utility class to train and fine-tune language models."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: torch.device | None = None,
        ignore_index: int = 0,
    ) -> None:
        """Initialise the trainer.

        Args:
            model: Model to optimise.
            learning_rate: Step size for the Adam optimiser.
            device: Optional ``torch`` device for computation.
            ignore_index: Target value excluded from cross-entropy. Generic
                datasets use ``0`` for padding; the paper path uses ``-100`` so
                predictive class ``0`` remains the real ``UNK`` class.
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def train(
        self,
        dataset: Dataset,
        epochs: int = 50,
        batch_size: int = 128,
        val_dataset: Optional[Dataset] = None,
        early_stopping: bool = False,
        patience: int = 5,
        start_epoch: int = 0,
        checkpoint_path: str | Path | None = None,
        show_progress: bool = True,
    ) -> None:
        self._ensure_trainable_embeddings()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        self.model.train()
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(start_epoch + 1, epochs + 1):
            total_loss = 0.0
            loop = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False) if show_progress else loader
            update_training_epoch(epoch)
            for x_batch, y_batch in loop:
                start_time = perf_counter()
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = self.criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                loss.backward()
                record_gradient_norm(self._gradient_norm())
                self.optimizer.step()
                total_loss += loss.item()
                if show_progress:
                    loop.set_postfix(loss=loss.item())
                observe_training_batch(int(x_batch.size(0)), perf_counter() - start_time)

            avg_loss = total_loss / len(loader)
            val_loss = self.evaluate(val_loader) if val_loader is not None else None
            if val_loss is not None:
                record_validation_loss(val_loss)
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
                    logger.info("Early stopping at epoch %s (best val loss %.4f)", epoch, best_val_loss)
                    break
            if checkpoint_path:
                self.save_checkpoint(checkpoint_path, epoch)

    def _ensure_trainable_embeddings(self) -> None:
        embed = getattr(self.model, "embed", None) or getattr(self.model, "embedding", None)
        weight = getattr(embed, "weight", None) if embed is not None else None
        if weight is None or not isinstance(weight, torch.Tensor) or not weight.requires_grad:
            return
        if torch.count_nonzero(weight).item() == 0:
            with torch.no_grad():
                weight.uniform_(-1e-3, 1e-3)

    def _gradient_norm(self) -> Optional[float]:
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return None
        total = torch.zeros(1, device=grads[0].device)
        for grad in grads:
            total += grad.data.norm(2).pow(2)
        return float(total.sqrt().item())

    def fine_tune(
        self,
        new_dataset: Dataset,
        epochs: int = 50,
        batch_size: int = 128,
        show_progress: bool = True,
    ) -> None:
        logger.info("Fine-tuning model on new data...")
        self.train(new_dataset, epochs=epochs, batch_size=batch_size, show_progress=show_progress)

    def evaluate(self, loader: DataLoader) -> float:
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

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        path = Path(path)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        start_time = perf_counter()
        torch.save(
            {"model_state": self.model.state_dict(), "optimizer_state": self.optimizer.state_dict(), "epoch": epoch},
            path,
        )
        observe_checkpoint(perf_counter() - start_time, epoch)

    def load_checkpoint(self, path: str | Path) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return int(checkpoint.get("epoch", 0))
